import numpy as np
import sys
from numba import jit

np.set_printoptions(threshold=sys.maxsize)
@jit(nopython=True)
def graph_shrinking(n, CAP, demand, edge_head, edge_tail, edge_value):
    ### note that component 0 is always the depot!!!
    n_customer = np.int64(n)
    n_component = np.int64(n+1)
    component_numbers = np.arange(n_customer + 1, dtype=np.int64)

    XMatrix = np.zeros((n_customer+1, n_customer+1), dtype=np.float64)
    node_idx = np.arange(n_customer+1, dtype=np.int64)

    for i in range(len(edge_value)):
        head = edge_head[i]
        tail = edge_tail[i]
        value = edge_value[i]
        XMatrix[head, tail] = value
        ######## x is an upper tirangular matrix.

    ####### PROPERLY INITIALIZE THESE ############
    depot_boundry = 0
    tot_demand = 0
    for i in range(1, n_customer+1):
        depot_boundry += XMatrix[0, i]
        tot_demand += demand[i]

    min_vehicles = np.ceil(tot_demand/CAP)
    QMin = tot_demand - CAP*(min_vehicles-1)
    UseDepotMatch = False

    slack = np.abs(2*min_vehicles - depot_boundry)
    if slack < 0.001: ########## precision check......
        UseDepotMatch = True

    shrink_again = True
    while shrink_again:
        shrink_again = False
        range_i = np.arange(1, n_component)
        np.random.shuffle(range_i)
        #################PAIR shrinking#####################
        for i in range_i:
            for j in range(i+1 ,n_component):
                if ((XMatrix[i, j] < 0.99) or (XMatrix[i, j] > 1.01)):
                    continue
                ##### SEARCH FOR A 1-edge#########
                for k in range(n_component):
                    if k == i or k == j:continue
                    edge_sum = XMatrix[i, k] + XMatrix[j, k] + XMatrix[k, i] + XMatrix[k, j]
                    if ((edge_sum < 0.99) or (edge_sum > 1.01)):
                        continue
                    if ((k != 0) or ((UseDepotMatch) and ((demand[i] + demand[j]) < QMin))):


                        old_comp_num_j = component_numbers[node_idx[j]]
                        old_comp_num_i = component_numbers[node_idx[i]]
                        new_comp_num = min(old_comp_num_j, old_comp_num_i)


                        for z in range(n_customer + 1):
                            if component_numbers[z] == old_comp_num_j or component_numbers[z] == old_comp_num_i:
                                component_numbers[z] = new_comp_num

                        demand[i] += demand[j]
                        demand[j] = 0

                        update_components(n_component, XMatrix, demand, node_idx, i, j)
                        shrink_again = True
                        n_component -= 1


                        break
                if shrink_again:
                    break
            if shrink_again:
                break
        ############## TRIPLET SHRINKING #####################
        p = np.random.rand(1)[0] ###### for randomization purposes
        if not shrink_again or p > 0.5:
            for i in range_i:
                for j in range(i+1 ,n_component):
                    if (XMatrix[i, j] < 0.001): continue
                    for l in range(j + 1, n_component):
                        if (XMatrix[i, l] < 0.001): continue
                        if (XMatrix[j, l] < 0.001): continue

                        edge_sum = XMatrix[i, j] + XMatrix[i, l] + XMatrix[j, l]

                        if ((edge_sum < 1.99) or (edge_sum > 2.01)):
                            continue

                        ############ GENERATE 1-edge ################
                        for k in range(n_component):
                            if k == i or k == j or k == l:
                                continue
                            edge_sum = (XMatrix[i, k] + XMatrix[j, k] + XMatrix[l, k] +
                                        XMatrix[k, i] + XMatrix[k, j] + XMatrix[k, l])
                            if ((edge_sum < 0.99) or (edge_sum > 1.01)):
                                continue
                            if ((k != 0) or ((UseDepotMatch) and ((demand[i] + demand[j] + demand[l]) < QMin))):


                                old_comp_num_j = component_numbers[node_idx[j]]
                                old_comp_num_l = component_numbers[node_idx[l]]
                                old_comp_num_i = component_numbers[node_idx[i]]

                                if (old_comp_num_j == old_comp_num_l): print("this cant happen man")

                                new_comp_num = min(old_comp_num_j, min(old_comp_num_i, old_comp_num_l))

                                for z in range(n_customer+1):
                                    if component_numbers[z] == old_comp_num_j or component_numbers[z] == old_comp_num_l or component_numbers[z] == old_comp_num_i:
                                        component_numbers[z] = new_comp_num

                                demand[i] += demand[j]
                                demand[i] += demand[l]
                                demand[j] = 0
                                demand[l] = 0

                                update_components(n_component, XMatrix, demand, node_idx, i, j, l=l)
                                n_component -= 2
                                shrink_again = True

                                break
                        if shrink_again:
                            break
                    if shrink_again:
                        break
                if shrink_again:
                    break

    new_head = np.zeros(len(edge_value))
    new_tail = np.zeros(len(edge_value))
    new_edge_value = np.zeros(len(edge_value))
    n_new_edges = 0
    for i in range(n_component):
        for j in range(i+1, n_component):
            if XMatrix[i, j]> 0:

                new_head[n_new_edges] = i
                new_tail[n_new_edges] = j
                new_edge_value[n_new_edges] = XMatrix[i, j]

                n_new_edges += 1

    return new_head, new_tail, new_edge_value, n_new_edges, n_component, demand, component_numbers, node_idx

@jit(nopython=True)
def update_components(n_components, XMatrix, demand, node_idx, i, j, l=-1):
    ######## taking care of the "pairwise combination"
    ######## outgoing arcs from j --> outgoing from i!
    XMatrix[i, j] = 0
    if l > 0:
        XMatrix[i, l] = 0
        XMatrix[j, l] = 0
    for k in range(j+1, n_components):
        XMatrix[i, k] += XMatrix[j, k]
        XMatrix[j, k] = 0
    ######## now handle the incoming arcs!!!!!!
    ######## incoming arcs: k-->j check if k>i or i>k
    for k in range(j):
        if i > k:
            XMatrix[k, i] += XMatrix[k, j]
        elif k > i:
            XMatrix[i, k] += XMatrix[k, j]

        XMatrix[k, j] = 0

    ###### same procedure with l.
    if l >0:
        for k in range(l + 1, n_components):
            XMatrix[i, k] += XMatrix[l, k]
            XMatrix[l, k] = 0

        for k in range(l):
            if i > k:
                XMatrix[k, i] += XMatrix[k, l]
            elif k > i:
                XMatrix[i, k] += XMatrix[k, l]

            XMatrix[k, l] = 0


    ####### move rows and columns: make the graph smaller!!!!!
    ## rows first:
    if l <0: l = n_components

    for row in range(j+1, l):
        demand[row - 1] = demand[row]
        node_idx[row - 1] = node_idx[row]
        demand[row] = 0
        node_idx[row] = -1
        for col in range(row+1, n_components):
            XMatrix[row-1, col] = XMatrix[row, col]
            XMatrix[row, col] = 0


    for col in range(j+1, l):
        for row in range(col):
            XMatrix[row, col-1] = XMatrix[row, col]
            XMatrix[row, col] = 0


    ###### this was j to l, now l to end: move 2 blocks!!!!!!

    if l < n_components-1:
        for row in range(l+1, n_components):
            demand[row - 2] = demand[row]
            node_idx[row - 2] = node_idx[row]
            demand[row] = 0
            node_idx[row] = -1
            for col in range(row+1, n_components):
                XMatrix[row-2, col] = XMatrix[row, col]
                XMatrix[row, col] = 0

        for col in range(l+1, n_components):
            for row in range(col):
                XMatrix[row, col-2] = XMatrix[row, col]
                XMatrix[row, col] = 0



