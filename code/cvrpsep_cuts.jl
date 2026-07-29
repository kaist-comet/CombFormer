using CVRPSEP

function str_comb(demand::Vector{Int},capacity::Int, edge_tail::Vector{Int}, 
    edge_head::Vector{Int}, 
    edge_x::Vector{Float64};)
    
    h, t, rhs = strengthened_comb_inequalities!(demand, capacity,edge_tail,edge_head,edge_x,max_n_cuts = 1000,
    Q_min = -1, 
    K = -1)

    return h,t,rhs
end

function rci(demand::Vector{Int},capacity::Int, edge_tail::Vector{Int}, 
    edge_head::Vector{Int}, 
    edge_x::Vector{Float64};)
    cm = CutManager()
    S, rhs = rounded_capacity_cuts!(cm, demand, capacity,edge_tail,edge_head,edge_x,max_n_cuts = min(length(demand), 100),  integrality_tolerance = 1e-3,)

    return S, rhs
end
