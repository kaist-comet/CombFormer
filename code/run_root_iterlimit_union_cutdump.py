#!/usr/bin/env python3
"""Root-node iteration-limited Union SCI experiment with cut dumps.

RCIs are separated every iteration; SCI separation starts after either
three consecutive low-RCI-violation rounds or 80% of the LP-iteration budget.
Only the root LP is solved. No branch-and-cut tree and no cut-pool management
are used here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import gurobipy as gp
import julia
import networkx as nx
import numpy as np
import torch
from GATRL import GATRL
from data_prep_rl_50_100_test import find_scis_nn


CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CVRP:
    def __init__(self, dimension, depot, demand, capacity, customers):
        self.dimension = dimension
        self.depot = depot
        self.demand = demand
        self.capacity = capacity
        self.customers = customers


def eucl_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_instance(n: int, seed: int, folder: Path) -> dict:
    path = folder / f"cvrp_n{n}_s{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing instance file: {path}")
    return json.loads(path.read_text())


def edge(i, j):
    return (i, j) if i < j else (j, i)


def between(source, target):
    rows = set()
    for i in source:
        for j in target:
            if i != j:
                rows.add(edge(i, j))
    return list(rows)


def complement(nodes, node_num):
    return list(set(range(node_num)) - set(nodes))


def complement_rci(nodes, node_num):
    return list(set(range(1, node_num)) - set(nodes))


def inner(nodes):
    rows = set()
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            rows.add(edge(nodes[i], nodes[j]))
    return list(rows)


def rounded_capacity_rhs(nodes, cvrp):
    return np.ceil(np.sum([cvrp.demand[i] for i in nodes]) / cvrp.capacity)


def sci_lhs_edges(handle, teeth, node_num):
    out_arcs = between(handle, complement(handle, node_num))
    for tooth in teeth:
        out_arcs.extend(between(tooth, complement(tooth, node_num)))
    return out_arcs


def cut_lhs(edge_values, arcs):
    return sum(edge_values.get(edge(*arc), 0.0) for arc in arcs)


def canonical_handle(handle):
    return tuple(sorted(set(int(v) for v in handle)))


def normalize_cvrpsep_scis(handles, teeth, rhs_values):
    normalized = []
    for handle, tooth_set, rhs in zip(handles, teeth, rhs_values):
        h = (np.array(handle).flatten() - 1).astype(int).tolist()
        t_norm = []
        for tooth in tooth_set:
            t = (np.array(tooth).flatten() - 1).astype(int).tolist()
            t_norm.append(t)
        normalized.append({"handle": h, "teeth": t_norm, "rhs": float(rhs)})
    return normalized


def add_rcis(model, edge_values_list, x, edge_head, edge_tail, jl):
    cvrp = model._cvrp
    node_num = len(cvrp.demand)
    list_s, list_rhs = jl.rci(
        cvrp.demand.astype(np.int64),
        cvrp.capacity,
        edge_tail,
        edge_head,
        edge_values_list,
    )

    max_violation = 0.0
    for subset, rhs in zip(list_s, list_rhs):
        subset = (np.array(subset).flatten() - 1).astype(int).tolist()
        if len(subset) > node_num / 2:
            subset_bar = complement_rci(subset, node_num)
            if len(subset_bar) == 0:
                model.addConstr(gp.quicksum(x[e] for e in inner(subset)) <= rhs)
            else:
                model.addConstr(
                    gp.quicksum(x[e] for e in inner(subset_bar))
                    + 0.5 * gp.quicksum(x[e] for e in between([0], subset_bar))
                    - 0.5 * gp.quicksum(x[e] for e in between([0], subset))
                    <= len(subset_bar) - rounded_capacity_rhs(subset, cvrp)
                )
        else:
            model.addConstr(gp.quicksum(x[e] for e in inner(subset)) <= rhs)

        violation = sum(x[e].x for e in inner(subset)) - rhs
        max_violation = max(max_violation, violation)

    return len(list_s), max_violation


def dump_event(out, event):
    out.write(json.dumps(event, separators=(",", ":")) + "\n")


def add_union_scis_with_dump(
    model,
    net,
    edge_values,
    edge_values_list,
    x,
    edge_head,
    edge_tail,
    tour_edges,
    n_rolls,
    jl,
    dump_out,
    n,
    seed,
    iteration,
    lb_before,
):
    cvrp = model._cvrp
    node_num = len(cvrp.demand)

    viols_nn, handles_nn, teeth_nn, rhs_nn = find_scis_nn(
        net,
        node_num,
        cvrp.demand,
        tour_edges,
        edge_values_list,
        Q=cvrp.capacity,
        n_shrink=15,
        n_rolls=n_rolls,
    )
    nn_scis = [
        {
            "handle": [int(v) for v in handle],
            "teeth": [[int(v) for v in tooth] for tooth in teeth],
            "rhs": float(rhs),
            "nn_violation": float(viol),
        }
        for handle, teeth, rhs, viol in zip(handles_nn, teeth_nn, rhs_nn, viols_nn)
    ]

    handles_cv, teeth_cv, rhs_cv = jl.str_comb(
        cvrp.demand.astype(np.int64),
        cvrp.capacity,
        edge_tail,
        edge_head,
        edge_values_list,
    )
    cvrpsep_scis = normalize_cvrpsep_scis(handles_cv, teeth_cv, rhs_cv)

    unique = {}
    raw_records = []
    for source, cuts in (("cvrpsep", cvrpsep_scis), ("combformer", nn_scis)):
        for raw_index, cut in enumerate(cuts):
            handle = cut["handle"]
            teeth = cut["teeth"]
            rhs = float(cut["rhs"])
            arcs = sci_lhs_edges(handle, teeth, node_num)
            lhs = cut_lhs(edge_values, arcs)
            violation = rhs - lhs
            key = canonical_handle(handle)
            raw_record = {
                "event": "raw_sci",
                "instance": f"cvrp_n{n}_s{seed}",
                "n": n,
                "seed": seed,
                "iteration": iteration,
                "lp_bound_before": lb_before,
                "source": source,
                "raw_index": raw_index,
                "handle": handle,
                "teeth": teeth,
                "rhs": rhs,
                "lhs": lhs,
                "violation": violation,
                "handle_size": len(set(handle)),
                "handle_demand": int(sum(cvrp.demand[v] for v in set(handle))),
                "handle_demand_ratio": float(
                    sum(cvrp.demand[v] for v in set(handle)) / cvrp.capacity
                ),
                "tooth_count": len(teeth),
                "avg_tooth_size": (
                    float(sum(len(set(tooth)) for tooth in teeth) / len(teeth))
                    if teeth
                    else 0.0
                ),
            }
            if source == "combformer":
                raw_record["nn_violation"] = float(cut.get("nn_violation", violation))
            raw_records.append(raw_record)

            rec = unique.get(key)
            if rec is None:
                unique[key] = {
                    "sources": {source},
                    "handle": handle,
                    "teeth": teeth,
                    "rhs": rhs,
                    "lhs": lhs,
                    "violation": violation,
                }
            else:
                rec["sources"].add(source)
                if rhs > rec["rhs"]:
                    rec.update(
                        {
                            "handle": handle,
                            "teeth": teeth,
                            "rhs": rhs,
                            "lhs": lhs,
                            "violation": violation,
                        }
                    )

    for record in raw_records:
        dump_event(dump_out, record)

    for key, rec in unique.items():
        handle = rec["handle"]
        teeth = rec["teeth"]
        rhs = rec["rhs"]
        arcs = sci_lhs_edges(handle, teeth, node_num)
        model.addConstr(gp.quicksum(x[e] for e in arcs) >= rhs)
        source_label = "+".join(sorted(rec["sources"]))
        dump_event(
            dump_out,
            {
                "event": "accepted_sci",
                "instance": f"cvrp_n{n}_s{seed}",
                "n": n,
                "seed": seed,
                "iteration": iteration,
                "lp_bound_before": lb_before,
                "source": source_label,
                "handle": handle,
                "teeth": teeth,
                "rhs": rhs,
                "lhs": rec["lhs"],
                "violation": rec["violation"],
                "handle_size": len(set(handle)),
                "handle_demand": int(sum(cvrp.demand[v] for v in set(handle))),
                "handle_demand_ratio": float(
                    sum(cvrp.demand[v] for v in set(handle)) / cvrp.capacity
                ),
                "tooth_count": len(teeth),
                "avg_tooth_size": (
                    float(sum(len(set(tooth)) for tooth in teeth) / len(teeth))
                    if teeth
                    else 0.0
                ),
            },
        )

    nn_count = sum(1 for rec in unique.values() if "combformer" in rec["sources"])
    ratio = nn_count / len(unique) if unique else 0.0
    return len(unique), ratio, len(cvrpsep_scis), len(nn_scis)


def build_root_lp(n, seed, instance, capacity):
    depot = 0
    demands = np.array(instance["demands"])
    coords = np.array(instance["coords"])
    customers = list(range(1, n + 1))
    nodes = list(range(0, n + 1))
    q = {i: int(demands[i]) for i in customers}
    q[depot] = 0

    graph = nx.complete_graph(nodes)
    pos = {node: tuple(coords[node]) for node in nodes}
    for i, j in graph.edges:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        graph.edges[i, j]["length"] = eucl_dist(x1, y1, x2, y2)

    min_vehicles = math.ceil(demands.sum() / capacity)
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    x = model.addVars(graph.edges, vtype=gp.GRB.CONTINUOUS, lb=0, ub=2, name="x")
    model.setObjective(
        gp.quicksum(graph.edges[e]["length"] * x[e] for e in graph.edges),
        gp.GRB.MINIMIZE,
    )
    model.addConstrs(
        gp.quicksum(x[e] for e in graph.edges if e in graph.edges(i)) == 2
        for i in customers
    )
    model.addConstr(
        gp.quicksum(x[e] for e in graph.edges if e in graph.edges(depot))
        >= 2 * min_vehicles
    )
    model.addConstrs(x[e] <= 1 for e in graph.edges() if depot not in e)
    model.optimize()

    model._depot = depot
    model._G = graph
    model._q = q
    model._Q = capacity
    model._cvrp = CVRP(len(nodes), depot, demands, capacity, customers)
    return model, x, graph


def iteration_limit_for_n(n):
    if n < 400:
        return 200
    if n < 600:
        return 100
    return 50


def run_instance(args, jl, net, n, seed, dump_path):
    set_seed(seed + 1)
    instance = load_instance(n, seed, args.data_dir)
    model, x, graph = build_root_lp(n, seed, instance, args.capacity)

    root_start = time.time()
    max_iter = iteration_limit_for_n(n)
    cuts = 0
    sci_cuts = 0
    raw_cvrpsep = 0
    raw_combformer = 0
    sci_iterations = []
    nn_ratios = []
    lb_sequence = []
    prev = model.objVal
    no_imp = 0
    total_iter = 0
    phase2_started = False
    low_rci_count = 0

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("w") as dump_out:
        while cuts < args.total_cut_limit:
            edge_values = {e: x[e].x for e in graph.edges}
            tour_edges = [e for e in graph.edges if edge_values[e] > 0]
            edge_head = []
            edge_tail = []
            edge_values_list = []
            for e in tour_edges:
                head, tail = e
                edge_tail.append(tail + 1)
                edge_head.append(head + 1)
                edge_values_list.append(edge_values[e])

            rci_cuts, rci_max_violation = add_rcis(
                model, edge_values_list, x, edge_head, edge_tail, jl
            )
            cuts += rci_cuts
            if rci_max_violation > 0.5:
                low_rci_count = 0
            else:
                low_rci_count += 1
                if low_rci_count >= 3:
                    phase2_started = True
            if total_iter >= 0.8 * max_iter:
                phase2_started = True

            cut_sci = 0
            nn_ratio = 0.0
            if phase2_started:
                cut_sci, nn_ratio, raw_cv, raw_nn = add_union_scis_with_dump(
                    model,
                    net,
                    edge_values,
                    edge_values_list,
                    x,
                    edge_head,
                    edge_tail,
                    tour_edges,
                    n_rolls=args.n_rolls,
                    jl=jl,
                    dump_out=dump_out,
                    n=n,
                    seed=seed,
                    iteration=total_iter,
                    lb_before=model.objVal,
                )
                raw_cvrpsep += raw_cv
                raw_combformer += raw_nn
                sci_cuts += cut_sci
                cuts += cut_sci
                if cut_sci > 0:
                    sci_iterations.append(total_iter + 1)
                    nn_ratios.append(nn_ratio)

            dump_event(
                dump_out,
                {
                    "event": "iteration",
                    "instance": f"cvrp_n{n}_s{seed}",
                    "n": n,
                    "seed": seed,
                    "iteration": total_iter,
                    "rci_cuts": rci_cuts,
                    "rci_max_violation": rci_max_violation,
                    "phase2_started": phase2_started,
                    "accepted_sci_cuts": cut_sci,
                    "combformer_ratio": nn_ratio,
                    "lb_before_optimize": model.objVal,
                },
            )

            model.optimize()
            if model.status != gp.GRB.OPTIMAL:
                break
            new_bound = model.objVal
            lb_sequence.append(new_bound)
            if new_bound <= prev + 1e-4:
                no_imp += 1
                if no_imp >= 10:
                    break
            else:
                no_imp = 0

            total_iter += 1
            prev = new_bound
            if args.max_time > 0 and time.time() - root_start > args.max_time:
                break
            if max_iter > 0 and total_iter >= max_iter:
                break

    elapsed = time.time() - root_start
    return {
        "instance": f"cvrp_n{n}_s{seed}",
        "n": n,
        "seed": seed,
        "lb": model.objVal,
        "time_seconds": elapsed,
        "iterations": total_iter,
        "total_cuts": cuts,
        "sci_cuts": sci_cuts,
        "raw_cvrpsep_sci": raw_cvrpsep,
        "raw_combformer_sci": raw_combformer,
        "combformer_ratio": float(np.mean(nn_ratios)) if nn_ratios else 0.0,
        "sci_iterations": sci_iterations,
        "lb_sequence": lb_sequence,
    }


def write_summary(rows, output_dir):
    lines = [
        "# Root Iteration-Limited Union Cut Dump",
        "",
        "| Instance | n | LB | Time (s) | Iter | Total cuts | SCI cuts | Raw CVRPSEP SCI | Raw CombFormer SCI | CombFormer ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['n']} | {row['lb']:.6g} | "
            f"{row['time_seconds']:.6g} | {row['iterations']} | "
            f"{row['total_cuts']} | {row['sci_cuts']} | "
            f"{row['raw_cvrpsep_sci']} | {row['raw_combformer_sci']} | "
            f"{row['combformer_ratio']:.4f} |"
        )
    if rows:
        lines += [
            "",
            "## Averages",
            "",
            "| n | Instances | Avg. LB | Avg. time (s) | Avg. iter | Avg. SCI cuts | Avg. raw CVRPSEP SCI | Avg. raw CombFormer SCI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for n in sorted({row["n"] for row in rows}):
            subset = [row for row in rows if row["n"] == n]
            lines.append(
                f"| {n} | {len(subset)} | "
                f"{np.mean([row['lb'] for row in subset]):.6g} | "
                f"{np.mean([row['time_seconds'] for row in subset]):.6g} | "
                f"{np.mean([row['iterations'] for row in subset]):.6g} | "
                f"{np.mean([row['sci_cuts'] for row in subset]):.6g} | "
                f"{np.mean([row['raw_cvrpsep_sci'] for row in subset]):.6g} | "
                f"{np.mean([row['raw_combformer_sci'] for row in subset]):.6g} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--base-seed", type=int, default=10_000_000)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "cvrp_instances")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "root_iterlimit_union_cutdump",
    )
    parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "models" / "train_smaller_2_tan5.pt")
    parser.add_argument("--n-rolls", type=int, default=8)
    parser.add_argument("--max-time", type=float, default=7200.0)
    parser.add_argument("--total-cut-limit", type=int, default=1_000_000)
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuts_dir = args.output_dir / "cuts"
    cuts_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(CODE_DIR)
    jl = julia.Julia(compiled_modules=False)
    jl.eval("using PyCall")
    jl.include(str(CODE_DIR / "cvrpsep_cuts.jl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GATRL(2, 1, 64, 4, False, True, 8).to(device)
    params = torch.load(args.model_path, map_location=device)
    net.load_state_dict(params)
    net.eval()

    rows = []
    for n in args.sizes:
        for offset in range(args.count):
            seed = args.base_seed + offset
            print(f"Running n={n} seed={seed}", flush=True)
            dump_path = cuts_dir / f"cvrp_n{n}_s{seed}.cuts.jsonl"
            row = run_instance(args, jl, net, n, seed, dump_path)
            rows.append(row)
            write_summary(rows, args.output_dir)

    write_summary(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
