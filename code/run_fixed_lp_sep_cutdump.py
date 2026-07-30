#!/usr/bin/env python3
"""Offline SCI separation on fixed RCI-only root LP trajectories.

This script mirrors the experiment in Section 5.2 of the manuscript:

1. Solve the root LP.
2. Iterate with RCI separation only.
3. At each RCI-only LP solution, run SCI separators offline on the same
   fractional support graph.
4. Record the generated SCI candidates, but never add SCIs to the LP.

The output is intended for checking separation-strength statistics such as
average and maximum violation on fixed LP trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
import julia
import numpy as np
import torch
from GATRL import GATRL
from data_prep_rl_50_100_test import find_scis_nn
from run_root_iterlimit_union_cutdump import (
    CODE_DIR,
    ROOT_DIR,
    add_rcis,
    build_root_lp,
    cut_lhs,
    dump_event,
    iteration_limit_for_n,
    load_instance,
    normalize_cvrpsep_scis,
    sci_lhs_edges,
    set_seed,
)


def edge(i, j):
    return (i, j) if i < j else (j, i)


def raw_sci_records(model, net, jl, n, seed, iteration, edge_values, edge_values_list,
                    edge_tail, edge_head, tour_edges, n_rolls):
    cvrp = model._cvrp
    node_num = len(cvrp.demand)

    handles_cv, teeth_cv, rhs_cv = jl.str_comb(
        cvrp.demand.astype(np.int64),
        cvrp.capacity,
        edge_tail,
        edge_head,
        edge_values_list,
    )
    cvrpsep_scis = normalize_cvrpsep_scis(handles_cv, teeth_cv, rhs_cv)

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
    combformer_scis = [
        {
            "handle": [int(v) for v in handle],
            "teeth": [[int(v) for v in tooth] for tooth in teeth],
            "rhs": float(rhs),
            "nn_violation": float(viol),
        }
        for handle, teeth, rhs, viol in zip(handles_nn, teeth_nn, rhs_nn, viols_nn)
    ]

    records = []
    for source, cuts in (("cvrpsep", cvrpsep_scis), ("combformer", combformer_scis)):
        for raw_index, cut in enumerate(cuts):
            handle = cut["handle"]
            teeth = cut["teeth"]
            rhs = float(cut["rhs"])
            lhs = cut_lhs(edge_values, sci_lhs_edges(handle, teeth, node_num))
            violation = rhs - lhs
            rec = {
                "event": "raw_sci",
                "instance": f"cvrp_n{n}_s{seed}",
                "n": n,
                "seed": seed,
                "iteration": iteration,
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
                rec["nn_violation"] = float(cut.get("nn_violation", violation))
            records.append(rec)
    return records


def run_instance(args, jl, net, n, seed, dump_path):
    set_seed(seed + 1)
    instance = load_instance(n, seed, args.data_dir)
    model, x, graph = build_root_lp(n, seed, instance, args.capacity)

    max_iter = iteration_limit_for_n(n)
    root_start = time.time()
    cuts = 0
    prev = model.objVal
    no_imp = 0
    total_iter = 0
    graph_count = 0
    raw_counts = defaultdict(int)
    violations = defaultdict(list)
    graph_avg_violations = defaultdict(list)
    graph_max_violations = defaultdict(list)
    graph_counts = defaultdict(list)

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
            model.optimize()
            if model.status != gp.GRB.OPTIMAL:
                break
            new_bound = model.objVal
            if new_bound <= prev + 1e-4:
                no_imp += 1
                if no_imp >= 10:
                    break
            else:
                no_imp = 0
            total_iter += 1
            prev = new_bound

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

            records = raw_sci_records(
                model,
                net,
                jl,
                n,
                seed,
                total_iter - 1,
                edge_values,
                edge_values_list,
                edge_tail,
                edge_head,
                tour_edges,
                args.n_rolls,
            )
            by_source = defaultdict(list)
            for rec in records:
                dump_event(dump_out, rec)
                by_source[rec["source"]].append(float(rec["violation"]))
            for source in ("cvrpsep", "combformer"):
                vals = by_source[source]
                raw_counts[source] += len(vals)
                graph_counts[source].append(len(vals))
                if vals:
                    violations[source].extend(vals)
                    graph_avg_violations[source].append(float(np.mean(vals)))
                    graph_max_violations[source].append(float(np.max(vals)))
                else:
                    graph_avg_violations[source].append(0.0)
                    graph_max_violations[source].append(0.0)

            dump_event(
                dump_out,
                {
                    "event": "fixed_lp_iteration",
                    "instance": f"cvrp_n{n}_s{seed}",
                    "n": n,
                    "seed": seed,
                    "iteration": total_iter - 1,
                    "lp_bound": model.objVal,
                    "support_edges": len(tour_edges),
                    "rci_cuts": rci_cuts,
                    "rci_max_violation": rci_max_violation,
                    "raw_cvrpsep_sci": len(by_source["cvrpsep"]),
                    "raw_combformer_sci": len(by_source["combformer"]),
                },
            )
            graph_count += 1

            if args.max_time > 0 and time.time() - root_start > args.max_time:
                break
            if max_iter > 0 and total_iter >= max_iter:
                break

    row = {
        "instance": f"cvrp_n{n}_s{seed}",
        "n": n,
        "seed": seed,
        "lb": model.objVal,
        "time_seconds": time.time() - root_start,
        "iterations": total_iter,
        "fixed_lp_graphs": graph_count,
        "rci_cuts": cuts,
    }
    for source in ("cvrpsep", "combformer"):
        vals = violations[source]
        row[f"{source}_raw_sci"] = raw_counts[source]
        row[f"{source}_avg_graph_avg_violation"] = float(np.mean(graph_avg_violations[source]))
        row[f"{source}_avg_graph_max_violation"] = float(np.mean(graph_max_violations[source]))
        row[f"{source}_avg_sci_per_graph"] = float(np.mean(graph_counts[source]))
        row[f"{source}_pooled_avg_violation"] = float(np.mean(vals)) if vals else 0.0
        row[f"{source}_pooled_max_violation"] = float(np.max(vals)) if vals else 0.0
    return row


def write_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed-LP Offline SCI Separation Dump",
        "",
        "This run evaluates SCI separators offline on RCI-only root LP trajectories.",
        "SCIs are not added to the LP.",
        "",
        "## Instance Results",
        "",
        "| Instance | n | Iter | Graphs | LB | Time (s) | CV avg viol | CV max viol | CV #SCI | CF avg viol | CF max viol | CF #SCI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['n']} | {row['iterations']} | "
            f"{row['fixed_lp_graphs']} | {row['lb']:.6g} | {row['time_seconds']:.6g} | "
            f"{row['cvrpsep_avg_graph_avg_violation']:.6g} | "
            f"{row['cvrpsep_avg_graph_max_violation']:.6g} | "
            f"{row['cvrpsep_avg_sci_per_graph']:.6g} | "
            f"{row['combformer_avg_graph_avg_violation']:.6g} | "
            f"{row['combformer_avg_graph_max_violation']:.6g} | "
            f"{row['combformer_avg_sci_per_graph']:.6g} |"
        )
    if rows:
        lines += [
            "",
            "## Averages",
            "",
            "| n | Instances | Avg. LB | Avg. time (s) | CV avg viol | CV max viol | CV #SCI | CF avg viol | CF max viol | CF #SCI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for n in sorted({row["n"] for row in rows}):
            sub = [row for row in rows if row["n"] == n]
            mean = lambda key: float(np.mean([row[key] for row in sub]))
            lines.append(
                f"| {n} | {len(sub)} | {mean('lb'):.6g} | {mean('time_seconds'):.6g} | "
                f"{mean('cvrpsep_avg_graph_avg_violation'):.6g} | "
                f"{mean('cvrpsep_avg_graph_max_violation'):.6g} | "
                f"{mean('cvrpsep_avg_sci_per_graph'):.6g} | "
                f"{mean('combformer_avg_graph_avg_violation'):.6g} | "
                f"{mean('combformer_avg_graph_max_violation'):.6g} | "
                f"{mean('combformer_avg_sci_per_graph'):.6g} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50])
    parser.add_argument("--base-seed", type=int, default=10_000_000)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "cvrp_instances")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "fixed_lp_sep_cutdump",
    )
    parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "models" / "train_smaller_2_tan5.pt")
    parser.add_argument("--n-rolls", type=int, default=8)
    parser.add_argument("--max-time", type=float, default=7200.0)
    parser.add_argument("--total-cut-limit", type=int, default=1_000_000)
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = args.model_path.resolve()
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
            print(f"Running fixed-LP separation n={n} seed={seed}", flush=True)
            dump_path = cuts_dir / f"cvrp_n{n}_s{seed}.cuts.jsonl"
            row = run_instance(args, jl, net, n, seed, dump_path)
            rows.append(row)
            write_summary(rows, args.output_dir)
    write_summary(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
