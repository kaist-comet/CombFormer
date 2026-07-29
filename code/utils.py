import math
import pickle
import numpy
"""
Python: Depot = 0 and customers are 1,....,n. The graph size is n+1 including the depot
Julia: Depot = 1 and customers are 2,.....,n+1.
CVRPSEP: Customers:1,......,n and Depot = n+1

Training Data (python format:)
Data = {"Locations": locations, "Weights": xvals, "Handles": handles_all, "Teeth": teeth_all, "Demands": demands, "Edges": edges}
Edges: Non-zero edges
Weights: Support graphs
Demands: Including zero --> Depot
"""


def complement(S, node_num):
    all_nodes = set(range(node_num))
    comp = all_nodes - set(S)
    return list(comp)
    

def delta_weight_sum(edges, S, xvals):
    if isinstance(S, int):
        # Handle case where S is an integer (treat it as i)
        i = S
        sm = 0
        for e in edges:
            if i in e:
                sm += xvals[tuple(e)]
        return sm

    elif isinstance(S, list):
        # Handle case where S is a set
        sm = 0
        for e in edges:
            i, j = e
            if i in S:
                if j not in S:
                    sm += xvals[tuple(e)]
            else:
                if j in S:
                    sm += xvals[tuple(e)]
        return sm
    else:
        raise ValueError("Input 'S' must be an integer or a set.")


def set_intersect(tooth, handle):
    if isinstance(tooth, int):
        int_list = []
        if tooth in handle:
            int_list.append(tooth)
        return int_list
    elif isinstance(tooth, list):
        int_list = []
        for t in tooth:
            if t in handle:
                int_list.append(t)
        return int_list
    else:
        raise ValueError("Input 'tooth' must be an integer or a set.")


def set_diff(tooth, handle):
    if isinstance(tooth, int):
        diff_list = []
        if tooth not in handle:
            diff_list.append(tooth)
        return diff_list
    elif isinstance(tooth, list):
        diff_list = []
        for t in tooth:
            if t not in handle:
                diff_list.append(t)
        return diff_list
    else:
        raise ValueError("Input 'tooth' must be an integer or a set.")
    
    
#n must be the number of customers + depot here!!
def r_tilde_to_r(s, n):
    if 0 not in s:
        return s
    else:
        return complement(s, n)
    

#n must be the number of customers + depot here!!
def r_tilde(s, n, demands, Q):
    s = r_tilde_to_r(s, n)
    dem = 0
    for i in s:
        dem += demands[i]
    return math.ceil(dem/Q)


def strccomb_lhs(edges, xvals, handle, teeth):
    lhs = delta_weight_sum(edges, handle, xvals)
    for tooth in teeth:
        lhs += delta_weight_sum(edges, tooth, xvals)
    return lhs


def strccomb_rhs(Q, demands, handle, teeth):
    rhs = 0
    n = len(demands) # make sure n =  cumber of customers + 1(depot)
    for t in teeth:
        rhs += r_tilde(set_intersect(t, handle), n, demands, Q)
        rhs += r_tilde(set_diff(t, handle), n, demands, Q)
        rhs += r_tilde(t, n, demands, Q)
    if rhs%2 == 1: # If odd +1: every vehicle entering a set must leave.
        rhs += 1
    return rhs


#If violation returns negative: NO VIOLATION, positive: the violation amount.
def strcomb_violation(Q, edges, xvals, demands, handle, teeth):
    ret = strccomb_lhs(edges, xvals, handle, teeth) - strccomb_rhs(Q, demands, handle, teeth)
    return -ret

#If violation returns negative: NO VIOLATION, positive: the violation amount.
# below is the violations of all handle&teeth pairs in a graph.
def strcomb_violation_graph(Q, edges, weights, demands, handle_b, teeth_b):
    xvals = {edges[i]: weights[i] for i in range(len(edges))}
    bs = len(handle_b)
    viols = []
    for i in range(bs):
        v = strcomb_violation(Q, edges, xvals, demands, handle_b[i], teeth_b[i])
        viols.append(v)
    return viols

'''
with open('new_cvrp_sep_100_capacity500_demand_random_integer_1to100_depot_is_0_1.pickle', 'rb') as handle:
    data = pickle.load(handle)

n = 100
Q = 500
edges_all = data["Edges"]
handles_all = data["Handles"]
teeth_all = data["Teeth"]
weights_all = data["Weights"]
demands_all = data["Demands"]
locations_all = data["Locations"]


for i in range(200):
    print(strcomb_violation_graph(Q, edges_all[i], weights_all[i], demands_all[i], handles_all[i], teeth_all[i]))

i = 200
pos = locations_all[i]
edge_list = edges_all[i]
handle = handles_all[i][0]
teeth = teeth_all[i][0]
teeth = [t for row in teeth for t in row]
nodes = set.union(set(handle),set(teeth))
ts = teeth_all[i][0]
nteeth = len(ts)
colors = ['pink', 'green', 'blue', 'purple', 'magenta', 'cyan']
G = nx.Graph(edge_list)


gg = G.subgraph(nodes=list(nodes))


belongs_to = []
for i in teeth:
    bs = []
    for j in range(nteeth):
        if i in ts[j]:
            bs.append(j)
    if i in handle:
        bs.append(nteeth)
    belongs_to.append(bs)


node_colors = []
for i in gg.nodes():
    if i in teeth:
        id = teeth.index(i)
        cols = [colors[b] for b in belongs_to[id]]
        n_col = len(cols)
        col_rat = 1/n_col
        if n_col>1:
            color_mix = '|'.join(cols)
        else:
            color_mix = cols[0]
    else:
        color_mix = colors[nteeth]
    node_colors.append(color_mix)
    print(color_mix)

nx.draw(gg, pos=pos, with_labels=True)
plt.show()
'''
