##### Given handle, CVRPSEP teeth sep. heuristc ###############
import numpy as np
import pickle
from numba import jit


@jit(nopython=True)
def initial_teeth_two_match(n_customer, cap, H, InHandle, edge_head, edge_tail, edge_value, demand, qmin, dep_match):
    #H_boundary calculation
    H_boundary = 0.0

    # First, find edges crossing handle:
    edges = len(edge_value)
    edge_handle = -np.ones(edges, dtype=np.int64) # inside handle
    edge_teeth = -np.ones(edges, dtype=np.int64) # in tooth
    x_between = -np.ones(edges, dtype=np.float64)
    num_edges = 0
    for j in range(edges):
        if InHandle[edge_head[j]] != InHandle[edge_tail[j]]:
            H_boundary += edge_value[j]
            # do not use the following edges for teeth
            if edge_head[j] == 0:
                if demand[edge_head[j]] >= qmin or dep_match == 0: continue
            else:
                if demand[edge_head[j]] + demand[edge_tail[j]] >= cap: continue

            x_between[num_edges] = edge_value[j]
            if edge_head[j] in H:
                edge_handle[num_edges] = edge_head[j]
                edge_teeth[num_edges] = edge_tail[j]
            else:
                edge_handle[num_edges] = edge_tail[j]
                edge_teeth[num_edges] = edge_head[j]

            num_edges += 1

    # Get sorted indices in descending order
    sorted_indices = np.argsort(x_between[:num_edges])[::-1]

    NoOfTeeth = 0
    OddTeeth = 0 # 1 if odd, 0 else
    ToothXSum = -np.ones(num_edges, dtype=np.float64)
    ToothTail = -np.ones(num_edges, dtype=np.int64)
    ToothHead = -np.ones(num_edges, dtype=np.int64) #always in handle


    for i in range(num_edges):
        EdgeNr = sorted_indices[i]
        t = edge_teeth[EdgeNr]
        h = edge_handle[EdgeNr]
        Xval = x_between[EdgeNr]

        if NoOfTeeth >= 3 and Xval <= 0.5 and OddTeeth == 1:
            break

        if i == 0:
            ToothXSum[i] = Xval
        else:
            ToothXSum[i] = ToothXSum[i-1] + Xval

        NoOfTeeth += 1
        OddTeeth = 1 - OddTeeth

        # disler kesisebilir mi? paperda sadece depoda diyor ve depo handle icinde olmamali... ama check bulamadim. sonra bak
        ToothTail[i] = t
        ToothHead[i] = h

        if NoOfTeeth > 3 and OddTeeth == 1:
            if ToothXSum[i] <= ToothXSum[i-2] + 1:
                NoOfTeeth -= 2

    InTooth = np.zeros((n_customer+1, NoOfTeeth), dtype=np.bool_)
    teeth = np.zeros((NoOfTeeth, n_customer), dtype=np.int64)
    teeth_fal = 2*np.ones(NoOfTeeth, dtype=np.int64)

    for t in range(NoOfTeeth):
        Tail = ToothTail[t]
        Head = ToothHead[t]
        InTooth[Tail, t] = 1
        InTooth[Head, t] = 1
        teeth[t, 0] = min(Tail, Head)
        teeth[t, 1] = max(Tail, Head)

    return InTooth, teeth, teeth_fal, H_boundary

# yukarda depo ile ilgili bi durum avr miyid? kulanmadim hic

@jit(nopython=True)
def expand_tooth(n_customer, CAP, demand, support, support_fal, XMatrix, NodeBoundry, InHandle, this_tooth_nr, teeth, teeth_fal, InTooth):
    no_teeth = np.int64(len(teeth))
    Q = np.zeros(4, dtype=np.float64)
    QCalc = np.zeros(4, dtype=np.float64)
    QCap = np.ones(4, dtype=np.float64)*CAP
    QK = np.ones(4, dtype=np.int64)

    NewQ = np.zeros(4, dtype=np.float64)
    NewQCalc = np.zeros(4, dtype=np.float64)
    NewQCap = np.ones(4, dtype=np.float64)*CAP
    NewQK = np.ones(4, dtype=np.int64)

    TouchedTooth = np.zeros(no_teeth, dtype=np.int64)

    Reached = np.zeros(n_customer + 1, dtype=np.int64)
    Selectable = np.ones(n_customer + 1, dtype=np.int64)
    Selectable[0] = 0
    XValue = np.zeros(n_customer + 1, np.float64)

    customers = np.arange(1, n_customer+1, dtype=np.int64)
    this_tooth = teeth[this_tooth_nr]
    stage = teeth_fal[this_tooth_nr] ##Initial tooth size
    node_stage = np.ones(n_customer+1, dtype=np.int64)*(n_customer+2)
    Inset = np.copy(InTooth[:, this_tooth_nr])
    ###### fixed only the following line...
    for j in range(teeth_fal[this_tooth_nr]):
        k = this_tooth[j]
        if k >=0:
            node_stage[k] = stage
            Selectable[k] = 0

    TouchedTooth[this_tooth_nr] = 1

    for t in range(no_teeth):
        if t == this_tooth_nr:
            continue

        teeth_intersect = 0
        #bu none i ben ekledim code icin dogru mu bilemedim. False da olabilir...
        intersect_in_handle = None

        for j in range(teeth_fal[t]):
            k = teeth[t, j]
            if node_stage[k] <= stage:
                teeth_intersect = 1
                intersect_in_handle = InHandle[k]
                TouchedTooth[t] = 1
                break

        if teeth_intersect == 1:
            for j in range(teeth_fal[t]):
                k = teeth[t, j]
                if InHandle[k] != intersect_in_handle:
                    Selectable[k] = 0

        ## in case we don't want them to intersect in handle

        for j in range(teeth_fal[t]):
            k = teeth[t, j]
            if InHandle[k] == 1:
                Selectable[k] = 0

    # no intersection both inside and outside handle

    # customers 1 to n_customer, consider indexing! range(1, n_customer+1)
    tooth_delta = 0
    depot_in_tooth = Inset[0]

    for i in customers:
        for j in range(support_fal[i]):
            k = support[i, j]
            if k > i or k==0:
                XVal = XMatrix[i][k]

                if Inset[i] and k>0:
                    XValue[k] += XVal
                    Reached[k] = True

                if Inset[k]:
                    XValue[i] += XVal
                    if k > 0:
                        Reached[i] = True

                if Inset[i] != Inset[k]:
                    tooth_delta += XVal

    total_demand = round(sum(demand))

    for i in customers:
        if Inset[i]:
            if InHandle[i]:
                Q[1] += demand[i]
            else:
                Q[2] += demand[i]
            Q[3] += demand[i]

    for m in range(1, 4):
        QCalc[m] = Q[m]

    if depot_in_tooth:
        QCalc[2] = total_demand - Q[2]
        QCalc[3] = total_demand - Q[3]



    for m in range(1, 4):
        QCap[m] = CAP
        QK[m] = 1
        while QCap[m] < QCalc[m]:
            QCap[m] += CAP
            QK[m] += 1

    LHS = tooth_delta
    RHS = QK[1] + QK[2] + QK[3]


    slack = LHS - RHS


    node_list_size = 0
    node_list = np.zeros(n_customer+1, dtype=np.int64)
    for i in range(n_customer+1):
        if Inset[i]:
            node_list[node_list_size] = i
            node_list_size += 1


    best_slack = slack
    best_list_size = node_list_size

    expand = True


    while expand:
        ######### add the customer with max x value #########

        best_x = -1
        best_node = -1
        best_x_node = -1

        new_best_slack = 2*(n_customer+2)

        for i in customers:
            if Selectable[i] == 0: continue
            if Reached[i] == 0: continue

            if XValue[i] > best_x:
                best_x = XValue[i]
                best_x_node = i

            for m in range(1, 4):
                NewQ[m] = Q[m]

            if InHandle[i]:
                NewQ[1] += demand[i]
            else:

                NewQ[2] += demand[i]

            NewQ[3] += demand[i]

            for m in range(1, 4):
                NewQCalc[m] = NewQ[m]
                NewQCap[m] = QCap[m]
                NewQK[m] = QK[m]

            if depot_in_tooth:
                NewQCalc[2] = total_demand - NewQ[2]
                NewQCalc[3] = total_demand - NewQ[3]

            for j in range(1, 4):
                while NewQCap[j] < NewQCalc[j]:
                    NewQCap[j] += CAP
                    NewQK[j] += 1

                while NewQCap[j] - CAP >= NewQCalc[j]:
                    NewQCap[j] -= CAP
                    NewQK[j] -= 1

            new_tooth_delta = tooth_delta + (NodeBoundry[i] - (2.0 * XValue[i]))
            new_slack = new_tooth_delta - (NewQK[1] + NewQK[2] + NewQK[3])

            if new_slack < new_best_slack - 0.001:
                new_best_slack = new_slack
                best_node = i

        if best_node <0:
            best_node = best_x_node

        if best_node >= 0:
            best_x = XValue[best_node]

            ########### include best node ##########

            Inset[best_node] = 1
            Selectable[best_node] = 0
            node_list[node_list_size] = best_node
            node_list_size += 1

            if InHandle[best_node]:
                Q[1] += demand[best_node]
            else:
                Q[2] += demand[best_node]

            Q[3] += demand[best_node]

            for m in range(1, 4):
                QCalc[m] = Q[m]

            if depot_in_tooth:
                QCalc[2] = total_demand - Q[2]
                QCalc[3] = total_demand - Q[3]

            for m in range(1, 4):
                while QCap[m] < QCalc[m]:
                    QCap[m] += CAP
                    QK[m] += 1

                while QCap[m] - CAP >= QCalc[m]:
                    QCap[m] -= CAP
                    QK[m] -= 1

            tooth_delta += NodeBoundry[best_node] - 2*best_x
            slack = tooth_delta - QK[1] - QK[2] - QK[3]

            if slack < (best_slack - 0.001) or RHS% 2 == 0:
                if (QK[1] + QK[2] + QK[3]) % 2 == 1:  # Odd sum
                    best_list_size = node_list_size
                    best_slack = slack

                    LHS = tooth_delta
                    RHS = QK[1] + QK[2] + QK[3]

            for k in support[best_node, :support_fal[best_node]]:
                if k > 0:
                    XVal = XMatrix[best_node][k]
                    XValue[k] += XVal
                    Reached[k] = 1

            # Identify new non-selectable nodes
            for t in range(no_teeth):
                if t == this_tooth_nr:
                    continue
                if InTooth[best_node][t] == 0:
                    continue

                if TouchedTooth[t] == 1:
                    continue
                # We already have a node from tooth t in ThisTooth

                intersect_in_handle = InHandle[best_node]

                for j in range(teeth_fal[t]):
                    k = teeth[t, j]

                    if InHandle[k] != intersect_in_handle:
                        Selectable[k] = 0

                TouchedTooth[t] = 1

        if best_node == -1:
            break

    t_new = np.zeros(best_list_size, dtype=np.int64)
    for i in range(best_list_size):
        t_new[i] = node_list[i]

    return RHS, LHS, t_new

@jit(nopython=True)
def expand_tooth_two_ways(n_customer, CAP, demand, support, support_fal, XMatrix, NodeBoundry, InHandle, this_tooth_nr, teeth, teeth_fal, InTooth):
    depot_in_tooth = InTooth[0][this_tooth_nr]

    global_rhs, global_lhs, best_list = expand_tooth(n_customer, CAP, demand, support, support_fal, XMatrix, NodeBoundry, InHandle, this_tooth_nr,
                                                teeth, teeth_fal, InTooth)

    best_slack = global_lhs - global_rhs

    if depot_in_tooth == 0:
        add_depot = 1

        for t in range(len(teeth)):
            if t == this_tooth_nr:
                continue
            if InTooth[0][t] == 0:
                continue
            ###### the depot is in tooth t ######
            ###### if this tooth and t intersects in handle, we cannot add depot to this tooth #####

            for j in range(teeth_fal[t]):
                k = teeth[t, j]
                if InTooth[k][this_tooth_nr]:
                    add_depot = 0
                    break

        if add_depot == 1:
            InTooth[0][this_tooth_nr] = 1

            teeth[this_tooth_nr, teeth_fal[this_tooth_nr]] = 0
            teeth_fal[this_tooth_nr] += 1

            rhs, lhs, candidate_best = expand_tooth(n_customer, CAP, demand, support, support_fal, XMatrix, NodeBoundry, InHandle, this_tooth_nr,
                                          teeth, teeth_fal, InTooth)


            slack = lhs - rhs

            if rhs % 2 ==1:
                if (slack < best_slack - 0.01) or (global_rhs % 2 == 0):

                    best_slack = slack

                    global_lhs = lhs
                    global_rhs = rhs

                    best_list = candidate_best

        ##### depot added #######

    for i in range(n_customer+1):
        InTooth[i][this_tooth_nr] = 0

    for i in range(len(best_list)):
        j = best_list[i]
        InTooth[j][this_tooth_nr] = 1

        teeth[this_tooth_nr, i] = best_list[i]
    teeth_fal[this_tooth_nr] = np.int64(len(best_list))

    return global_lhs, global_rhs, InTooth, teeth, teeth_fal


def main_teeth_sep(n_customer, H, CAP, demand, edge_head, edge_tail, edge_value):
    n_customer = np.int64(n_customer)
    H = np.array(H, dtype=np.int64)
    edge_head = np.array(edge_head, dtype=np.int64)
    edge_tail = np.array(edge_tail, dtype=np.int64)
    CAP = np.int64(CAP)
    demand = np.array(demand, dtype=np.int64)
    edge_value = np.array(edge_value).astype(np.float64)
    InHandle = np.zeros(n_customer+1, dtype=np.uint8)
    NodeBoundry= np.zeros(n_customer+1, dtype=np.float64)

    support = -np.ones((n_customer+1, n_customer+1), dtype=np.int64)
    support_fal = np.ones(n_customer+1, dtype=np.int64)

    total_demand = demand.sum()
    min_vehicles = np.ceil(total_demand / CAP).astype(np.int64)
    qmin = total_demand - (min_vehicles - 1) * CAP
    dep_boundry = 0.0


    for i in range(n_customer+1):
        if i in H:
            InHandle[i] = 1

    XMatrix = np.zeros((n_customer+1, n_customer+1), dtype=np.float64)

    for i in range(len(edge_value)):
        head = edge_head[i]
        tail = edge_tail[i]
        value = edge_value[i]
        if head == 0 or tail == 0:
            dep_boundry += value

        XMatrix[head, tail] = value
        XMatrix[tail, head] = value
        NodeBoundry[tail] += value
        NodeBoundry[head] += value

        if tail not in support[head, :support_fal[head]]:
            support[head, support_fal[head]] = tail
            support_fal[head] += 1

        if head not in support[tail, :support_fal[tail]]:
            support[tail, support_fal[tail]] = head
            support_fal[tail] += 1

    depot_match = 0
    slack = np.abs(2*min_vehicles - dep_boundry)
    if slack < 0.001: ########## precision check......
        depot_match = 1

    in_tooth, teeth, teeth_fal, h_boundary = initial_teeth_two_match(n_customer, CAP, H, InHandle, edge_head, edge_tail, edge_value, demand, qmin, depot_match)

    ToothSurplusSum = 0.0
    LHSSum = h_boundary
    RHSSum = 0


    for t in range(len(teeth)):
        t = np.int64(t)
        t_lhs, t_rhs, in_tooth, teeth, teeth_fal = expand_tooth_two_ways(n_customer, CAP, demand, support, support_fal, XMatrix, NodeBoundry, InHandle, t, teeth, teeth_fal, in_tooth)
        LHSSum += t_lhs
        RHSSum += t_rhs

        tooth_surplus = t_rhs - t_lhs
        ToothSurplusSum += tooth_surplus

    viol = LHSSum - RHSSum - RHSSum%2
    r = RHSSum + RHSSum % 2
    if len(teeth)<2:
        viol = 0
    teeth = np.asarray(teeth)
    teeth_fal = np.asarray(teeth_fal)
    t = (teeth, teeth_fal)

    return t, viol, r
