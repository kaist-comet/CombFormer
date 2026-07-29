from torch_geometric.data import Data, InMemoryDataset
import torch
import numpy as np
from graph_shrinking import graph_shrinking
from teethsep_valid import main_teeth_sep
from torch_geometric.data import Batch
import copy
from collections import Counter

def compute_handle_uniqueness_all(handles):
    """
    Efficient batch computation of uniqueness for all handles.

    Parameters:
    ----------
    handles : List[List[int]]
        List of all handles, using global node indices.

    Returns:
    -------
    List[float]
        Uniqueness score per handle.
        One value per handle: average handle count for its nodes.

    avg participation metric is more sensitive to partial overlap, while uniqueness score is a binary count per node.
    """
    # Flatten all node IDs across handles and count occurrences
    all_nodes = [node for h in handles for node in h]
    node_counts = Counter(all_nodes)

    uniqueness_scores = np.zeros(len(handles))
    avg_participation = np.zeros(len(handles))
    idx = -1
    for h in handles:
        idx += 1
        unique_count = sum(1 for node in h if node_counts[node] == 1)
        uniqueness_scores[idx] = unique_count / len(h)

        total = sum(node_counts[node] for node in h)
        avg_participation[idx] = total / len(h)

    ## scaled (0, 1]. 1 means every node is unique to this handle. the smaller the more overlap with other handles.

    diversity = 1/avg_participation

    return uniqueness_scores, avg_participation, diversity


def violation_ngs(data_g, handles):
    data = copy.deepcopy(data_g).cpu()
    graph_mask = (data.batch == 0)
    edge_mask = graph_mask[data.edge_index[0]] & graph_mask[data.edge_index[1]]
    graph_edges = data.edge_index[:, edge_mask]
    graph_weights = data.edge_attr[edge_mask]
    graph_demands = data.demand[graph_mask]

    # Identify the depot node (smallest index in the graph)
    graph_nodes = graph_mask.nonzero(as_tuple=False).view(-1)
    n = len(graph_nodes) - 1
    depot = graph_nodes[0]  # Assumes the depot is the smallest index

    edge_index = graph_edges.numpy() - depot.item()
    edge_value = graph_weights.numpy()
    demand = graph_demands.numpy()
    heads = edge_index[0, :]
    tails = edge_index[1, :]
    n = data.n[0]
    n_o = data.n_original[0]
    violations = []
    hs = []
    ts = []
    rhs_s = []
    if n_o > n:
        # shrunk graph
        c_nrs = data.component_nrs[0]
        n_idx = data.node_idx[0]
        for i in range(len(handles)):
            handle = list(handles[i])
            teeth, viol, rhs = main_teeth_sep(n, handle, data.Q[0].item(), demand, heads, tails, edge_value)
            t, tfal = teeth[0], teeth[1]
            t_list = []
            for k in range(len(tfal)):
                t_list.append(t[k, :tfal[k]].tolist())
            violations.append(-viol)
            handle_o, teeth_o, comps = project_to_original_graph(n, c_nrs, n_idx, handle, t_list)
            ts.append(teeth_o)
            hs.append(handle_o)
            rhs_s.append(rhs)
    else:
        for i in range(len(handles)):
            handle = list(handles[i])
            teeth, viol, rhs = main_teeth_sep(n, handle, data.Q[0].item(), demand, heads, tails, edge_value)
            t, tfal = teeth[0], teeth[1]
            t_list = []
            for k in range(len(tfal)):
                t_list.append(t[k, :tfal[k]].tolist())
            violations.append(-viol)
            hs.append(handle)
            ts.append(t_list)
            rhs_s.append(rhs)
    return violations, hs, ts, rhs_s


class NGS():
    def __init__(self, net, n_pop, n_off, eval, n_gen, k=1, mu=0.5):
        self.net = net
        self.n_pop = n_pop
        self.n_off = n_off
        self.eval = eval
        self.mu = mu
        self.n_gen = n_gen
        self.k = k

    def initial_generation(self, data, data_large):
        self.net.encode_init(data)
        handles, handles_dict = self.net.generation_zero(data, data_large, mu=self.mu)
        return handles, handles_dict

    def rank(self, rewards):
        ranks = np.argsort(rewards)  # Sort rewards (ascending order: lower is worse)
        # Compute probabilities based on the given formula
        probs = 1 / (self.k * len(rewards) + ranks)
        # Normalize to sum to 1
        probs /= probs.sum()
        return probs

    def run(self, data, data_large):
        handles, handles_dict = self.initial_generation(data, data_large)
        rewards, hs, ts, rhs_s = self.eval(data, handles)
        # Track best handle & reward
        best_handle = None
        best_reward = float("-inf")

        # Store initial rewards
        idx = 0
        for h, r in zip(handles, rewards):
            handles_dict[h]["reward"] = r
            handles_dict[h]["original"] = hs[idx]
            handles_dict[h]["teeth"] = ts[idx]
            handles_dict[h]["rhs"] = rhs_s[idx]
            idx += 1
            if r < 0 and len(handles_dict) > 4:
                del handles_dict[h]
            else:
                if r > best_reward:
                    best_reward = r
                    best_handle = h

        for i in range(self.n_gen):
            # 2️⃣ Rank & Select Two Parents
            rewards_list = [handles_dict[h]["reward"] for h in handles_dict]
            probs = self.rank(rewards_list)

            # Get Parent Handles & Combine Their Masks
            parent_keys = list(handles_dict.keys())  # Get handles in order
            # Determine how many pairs we can afford without replacement
            max_no_replacement_pairs = len(handles_dict) // 2  # Max unique pairs possible
            num_pairs = self.n_off // 2  # Desired number of pairs
            no_replacement_pairs = min(num_pairs, max_no_replacement_pairs)
            # First, select as many as possible without replacement
            if max_no_replacement_pairs > 0:
                parent_pairs = np.random.choice(len(probs), size=(no_replacement_pairs, 2),
                                                replace=False, p=probs)

            # If more pairs are needed, select the remaining ones with replacement
            remaining_pairs = num_pairs - no_replacement_pairs
            if remaining_pairs > 0:
                parent_pairs_replacement = np.random.choice(len(probs), size=(remaining_pairs, 2), replace=True,
                                                            p=probs)
                if max_no_replacement_pairs > 0:
                    parent_pairs = np.vstack((parent_pairs_replacement, parent_pairs))
                else:
                    parent_pairs = parent_pairs_replacement

            # 3️⃣ Generate Offspring from Multiple Parent Pairs
            for p1_idx, p2_idx in parent_pairs:
                p1, p2 = handles_dict[parent_keys[p1_idx]]["mask"], handles_dict[parent_keys[p2_idx]]["mask"]
                parent_mask = p1 | p2  # Combine masks

                # Generate 2 offspring from this pair
                offspring_handles, offspring_masks = self.net.generation_next_off(data, parent_mask, mu=self.mu)

                # 4️⃣ Evaluate Only Unique Offspring
                for h, m in zip(offspring_handles, offspring_masks):
                    if h not in handles_dict:  # Check if handle is unique
                        handles_dict[h] = {"mask": m, "reward": 0}  # Initialize reward as 0
                        rew, hs, ts, rhs_s = self.eval(data, [h])  # Expensive evaluation
                        r = rew[0]
                        # Track the best handle
                        if r < 0.001:
                            continue
                        handles_dict[h]["reward"] = r  # Store actual reward
                        handles_dict[h]["original"] = hs[0]
                        handles_dict[h]["teeth"] = ts[0]
                        handles_dict[h]["rhs"] = rhs_s[0]
                        if r > best_reward:
                            best_reward = r
                            best_handle = h

            # 5️⃣ Maintain Population Size (Stochastic Sampling)
            if len(handles_dict) > self.n_pop:
                # Get updated rewards
                rewards_list = [handles_dict[h]["reward"] for h in handles_dict]

                # Rank & Sample `self.n_pop` individuals
                probs = self.rank(rewards_list)
                selected_indices = set(np.random.choice(len(probs), size=self.n_pop, replace=False, p=probs))

                # Delete unselected handles **instead of recreating the dictionary**
                selected_keys = list(handles_dict.keys())
                for idx, key in enumerate(selected_keys):
                    if idx not in selected_indices:
                        del handles_dict[key]  # In-place deletion

        handles_ = []
        teeth_ = []
        rhs_ = []
        viols = []
        for h in handles_dict:
            if handles_dict[h]["reward"] > 0.01:
                handles_.append(handles_dict[h]["original"])
                teeth_.append(handles_dict[h]["teeth"])
                rhs_.append(handles_dict[h]["rhs"])
                viols.append(handles_dict[h]["reward"])

        return handles_, teeth_, rhs_, viols



def data_prep_ngs(n, demands, edges, edge_weights, Q=500, shrinking=False, n_roll=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    # create the list of Data instances. each instance is a graph
    # Node features: is_depot and demands/capacity
    dem_ = np.array(demands)
    is_dep = [0] * n
    is_dep[0] = 1
    x_ = [dem_ / Q, is_dep]
    x = torch.tensor(np.array(x_).T, dtype=torch.float, requires_grad=False, device=device)
    demand = torch.tensor(demands, dtype=torch.float, requires_grad=False, device=device)
    e_all = np.array(edges).T
    edge_values = np.array(edge_weights)
    edge_values = np.round(edge_values, 5)
    edge_index = torch.tensor(e_all, dtype=torch.long, requires_grad=False,
                              device=device)
    edge_weight = torch.tensor(edge_values, dtype=torch.float, requires_grad=False, device=device)

    if shrinking:
        dat = shrink(is_dep, n - 1, Q, dem_, e_all[0, :], e_all[1, :], edge_values, device)
    else:
        dat = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, demand=demand,
                   Q=500, n=n, n_original=n, node_idx=[], component_nrs=[]).to(device)
    data_all = Batch.from_data_list([dat] * n_roll)
    data_small = Batch.from_data_list([dat])
    return data_small, data_all

def validate(handle, teeth):
    n_teeth = len(teeth)
    h = set(handle)
    #code for n_teeth problem is 1
    if n_teeth <2:
        return (0,1)
    for i in range(n_teeth):
        t = set(teeth[i])
        if len(t.difference(h))==0:
            #code for the diff T/H is empty..is 2
            return (0,2)
        if len(t.intersection(h))==0:
            # they must intersenct if empty, code is 3
            return (0,3)
        for j in range(i+1,n_teeth):
            tj = set(teeth[j])
            t_int = t.intersection(tj)
            #code for intersection both in and out handle
            if len(t_int.intersection(h))>0:
                if len(t_int.difference(h))>0:
                    print("ISSSSSSSUUUUUUUEEEEEE")
                    print((0,4,t_int.intersection(h),t_int.difference(h)))
                    return (0,4,t_int.intersection(h),t_int.difference(h))
    return (1,0)


def violation_fn(data_g, handles):
    data = copy.deepcopy(data_g).cpu()
    batch_size = data.batch.max().item() + 1
    violations = []
    ts = []
    hs = []
    rhs = []
    for i in range(batch_size):
        handle = handles[i]
        graph_mask = (data.batch == i)
        edge_mask = graph_mask[data.edge_index[0]] & graph_mask[data.edge_index[1]]
        graph_edges = data.edge_index[:, edge_mask]
        graph_weights = data.edge_attr[edge_mask]
        graph_demands = data.demand[graph_mask]
        c_nrs = data.component_nrs[i]
        n_idx = data.node_idx[i]
        graph_nodes = graph_mask.nonzero(as_tuple=False).view(-1)
        depot = graph_nodes[0]  # Assumes the depot is the smallest index

        edge_index = graph_edges.numpy() - depot.item()

        edge_value = graph_weights.numpy()
        demand = graph_demands.numpy()
        heads = edge_index[0, :]
        tails = edge_index[1, :]
        n = data.n[i]
        teeth, viol, r = main_teeth_sep(n, handle, data.Q[0].item(), demand, heads, tails, edge_value)
        t, tfal = teeth[0], teeth[1]
        t_list = []
        for k in range(len(tfal)):
            t_list.append(t[k, :tfal[k]].tolist())

        if data.n_original[i] > data.n[i]:
            handle_o, teeth_o, comps = project_to_original_graph(data.n[i], c_nrs, n_idx, handle, t_list)
            ts.append(teeth_o)
            hs.append(handle_o)
            rhs.append(r)
        else:
            ts.append(t_list)
            hs.append(handle)
            rhs.append(r)
        violations.append(-viol)
    return violations, hs, ts, rhs


def project_to_original_graph(n_comp, c_nrs, n_idx, handle, teeth):
    n_idx = n_idx[:n_comp]
    comps_to_nodes = [[] for _ in range(n_comp)]
    for i in range(n_comp):
        original_node = n_idx[i]
        comp = c_nrs[original_node]
        for j in range(len(c_nrs)):
            if c_nrs[j] == comp:
                comps_to_nodes[i].append(j)

    handle_o = []
    for i in handle:
        handle_o += comps_to_nodes[i]

    teeth_o = [[] for i in range(len(teeth))]
    for t in range(len(teeth)):
        for node in teeth[t]:
            teeth_o[t] += comps_to_nodes[node]

    return handle_o, teeth_o, comps_to_nodes


def project_to_original_handles(comps_to_nodes, handle):
    handle_o = []
    for i in handle:
        handle_o += comps_to_nodes[i]
    return handle_o

def find_comp_nrs(n_comp, c_nrs, n_idx):
    n_idx = n_idx[:n_comp]
    comps_to_nodes = [[] for _ in range(n_comp)]
    for i in range(n_comp):
        original_node = n_idx[i]
        comp = c_nrs[original_node]
        for j in range(len(c_nrs)):
            if c_nrs[j] == comp:
                comps_to_nodes[i].append(j)
    return comps_to_nodes


def project_to_original_teeth(comps_to_nodes, teeth):
    teeth_o = [[] for i in range(len(teeth))]
    for t in range(len(teeth)):
        for node in teeth[t]:
            teeth_o[t] += comps_to_nodes[node]
    return teeth_o

def shrunk_data(is_dep, n_customer, Q, dd, head, tail, edge_values, d_small, i,  device):
    demand = dd.copy()
    new_head, new_tail, new_edge_value, n_new_edges, n_component, demand_new, component_nrs, node_idx = graph_shrinking(n_customer, Q, demand, head, tail, edge_values)
    x_ = [demand_new[:n_component] / Q, is_dep[:n_component]]
    x = torch.tensor(np.array(x_).T, dtype=torch.float, requires_grad=False, device=device)
    demand_new = torch.tensor(demand_new[:n_component], dtype=torch.float, requires_grad=False, device=device)
    edge_index = torch.tensor(np.array([new_head[:n_new_edges], new_tail[:n_new_edges]]), dtype=torch.long, requires_grad=False, device=device)
    edge_weight = torch.tensor(new_edge_value[:n_new_edges], dtype=torch.float, requires_grad=False, device=device)
    dat = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, demand=demand_new,
               Q=Q, n=n_component, n_original=n_customer+1, node_idx=node_idx, component_nrs=component_nrs).to(device)
    d_small[i] = dat


def shrink(is_dep, n_customer, Q, dd, head, tail, edge_values, device):
    demand = dd.copy()
    new_head, new_tail, new_edge_value, n_new_edges, n_component, demand_new, component_nrs, node_idx = graph_shrinking(n_customer, Q, demand, head, tail, edge_values)
    x_ = [demand_new[:n_component] / Q, is_dep[:n_component]]
    x = torch.tensor(np.array(x_).T, dtype=torch.float, requires_grad=False, device=device)
    demand_new = torch.tensor(demand_new[:n_component], dtype=torch.float, requires_grad=False, device=device)
    edge_index = torch.tensor(np.array([new_head[:n_new_edges], new_tail[:n_new_edges]]), dtype=torch.long, requires_grad=False, device=device)
    edge_weight = torch.tensor(new_edge_value[:n_new_edges], dtype=torch.float, requires_grad=False, device=device)
    dat = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, demand=demand_new,
               Q=Q, n=n_component, n_original=n_customer+1, node_idx=node_idx, component_nrs=component_nrs).to(device)
    return dat


def data_prep_test(n, demands, edges, edge_weights, Q=500, n_shrink=0, n_roll=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    # create the list of Data instances. each instance is a graph
    # Node features: is_depot and demands/capacity
    dem_ = np.array(demands)
    is_dep = [0] * n
    is_dep[0] = 1
    x_ = [dem_ / Q, is_dep]
    x = torch.tensor(np.array(x_).T, dtype=torch.float, requires_grad=False, device=device)
    demand = torch.tensor(demands, dtype=torch.float, requires_grad=False, device=device)
    e_all = np.array(edges).T
    edge_values = np.array(edge_weights)
    edge_values = np.round(edge_values, 5)
    edge_index = torch.tensor(e_all, dtype=torch.long, requires_grad=False,
                              device=device)
    edge_weight = torch.tensor(edge_values, dtype=torch.float, requires_grad=False, device=device)

    dat = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, demand=demand,
               Q=500, n=n, n_original=n, node_idx=[], component_nrs=[]).to(device)
    # create 3 data from 1 graph
    data_list = [[] for _ in range((n_roll-1)*(n_shrink + 1))]
    data_list_small = [[] for _ in range(n_shrink + 1)]
    data_list_small[0] = dat
    for k in range(1, n_shrink+1):
        shrunk_data(is_dep, n - 1, Q, dem_, e_all[0, :], e_all[1, :], edge_values, data_list_small, k, device)

    incr = n_shrink + 1
    for j in range(n_roll-1):
        id_st = int(j*incr)
        id_end = int((j+1)*incr)
        data_list[id_st:id_end] = data_list_small
    if n_roll > 1:
        data_list = Batch.from_data_list(data_list)
    data_list_small = Batch.from_data_list(data_list_small)
    return data_list, data_list_small

def find_scis_nn(net, n, demands, edges, edge_weights, Q=500, n_shrink=3, n_rolls=16):
    net.eval()

    data_graph, data_graph_small = data_prep_test(n, demands, edges, edge_weights, Q=Q, n_shrink=n_shrink, n_roll=n_rolls)

    with torch.no_grad():
        if n_rolls>1:
            handls, _, _ = net.forward_b(data_graph, rollout=False)
            violations, handles, teeth_s, rhs_s = violation_fn(data_graph, handls)
        else:
            violations = []
            handles = []
            teeth_s = []
            rhs_s = []
        handle, _, _ = net.forward_b(data_graph_small, rollout=True)
        viols, hs, ts, rhs = violation_fn(data_graph_small, handle)

        violations.extend(viols)
        handles.extend(hs)
        teeth_s.extend(ts)
        rhs_s.extend(rhs)

        # Ensure unique handles (handling list of lists)
        unique_handles = []
        unique_violations = []
        unique_ts = []
        unique_rhs = []
        seen = set()
        for i in range(len(handles)):
            h = handles[i]
            h_tuple = tuple(sorted(h))  # Convert list to sorted tuple for uniqueness check
            if h_tuple not in seen and violations[i] > 0.01: #shoul i make this smaller?
                seen.add(h_tuple)
                unique_handles.append(h)  # Append original list format
                unique_violations.append(violations[i])
                unique_ts.append(teeth_s[i])
                unique_rhs.append(rhs_s[i])

    return unique_violations, unique_handles, unique_ts, unique_rhs



def find_scis_nn_ngs(net, n, demands, edges, edge_weights, Q=500, n_roll=31, shr=False):
    net.eval()
    data_graph, data_large = data_prep_ngs(n, demands, edges, edge_weights, Q=Q, shrinking=shr, n_roll=n_roll)
    with torch.no_grad():
        ngs = NGS(net, 64, 8, violation_ngs, 8, k=2, mu=0.5)
        handles_, teeth_, rhs_, viols = ngs.run(data_graph, data_large)
    return viols, handles_, teeth_, rhs_

