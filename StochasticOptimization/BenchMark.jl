using SparseArrays
include("MixedIntegerSolver.jl")

function two_stage_so_centralized(model_first_stage, model_second_stage, options)
# A centralize optimzation problem as a benchmark
    ns = length(model_second_stage)
    ps = model_first_stage["ps"]

    c = model_first_stage["c"]
    A = model_first_stage["A"]
    b = model_first_stage["b"]
    vtype_first_stage = model_first_stage["vtype"]
    lx = model_first_stage["lb"]
    ux = model_first_stage["ub"]
    NX = length(lx)

    Ts = []
    Ws = []
    hs = []
    ds = []
    ly = []
    uy = []
    con_obj = []
    vtype_second_stage = []
    for s=1:ns
        push!(Ts, model_second_stage[s]["T"])
        push!(Ws, model_second_stage[s]["W"])
        push!(hs, model_second_stage[s]["hs"])
        push!(ds, model_second_stage[s]["d"])
        push!(ly, model_second_stage[s]["lb"])
        push!(uy, model_second_stage[s]["ub"])
        push!(vtype_second_stage, model_second_stage[s]["vtype"])
    end

    NY = length(ds[1])
    # Construct a dictionary 
    model_master = Dict("obj" => [c; ps], 
                        "lb" => [lx;zeros(ns,1)],
                        "ub" => [ux;Inf*ones(ns,1)],
                        "A" => hcat(A, spzeros(size(A,1), ns)),
                        "rhs" => b,
                        "sense" => model_first_stage["sense"],
                        "vtype" => [vtype_first_stage;repeat(["C"], outer = [ns,1])],
                        )

    model_centralized = model_master

    for s = 1:ns
        model_centralized["obj"] = [model_centralized["obj"]; zeros(NY,1)]
        model_centralized["lb"] = [model_centralized["lb"]; ly[s]]
        model_centralized["ub"] = [model_centralized["ub"]; uy[s]]
        model_centralized["A"] = [hcat(model_centralized["A"], spzeros(size(model_centralized["A"],1), NY)); 
                                  hcat(Ts[s], spzeros(size(Ts[s],1), ns + NY*(s-1)), Ws[s])]
        model_centralized["rhs"] = [model_centralized["rhs"]; hs[s]]
        model_centralized["sense"] = [model_centralized["sense"]; repeat([">"], outer = [length(hs[s]),1]) ]
        model_centralized["vtype"] = [model_centralized["vtype"]; vtype_second_stage[s]]
        temp = zeros(1, NX + ns + (s-1)*NY )
        temp[NX + s] = 1
        model_centralized["A"] = [model_centralized["A"]; hcat(temp, -ds[s]')]
        model_centralized["rhs"] = [model_centralized["rhs"]; 0]
        model_centralized["sense"] = [model_centralized["sense"]; ">"]
    end
    model_centralized["sense"] = model_centralized["sense"][:,1]
    model_centralized["vtype"] = model_centralized["vtype"][:,1]
    result = mixed_integer_linear_programming(model_centralized)

    x_best = result["x"][1:NX]
    y_best = [ ]
    obj_second_stage_best = zeros(ns,1)
    for s = 1 : ns
        push!(y_best, result["x"][NX + ns + (s-1)*NY +  1: NX + ns + s*NY])
        # print(typeof(ds[s]))
        # print(typeof(y_best[s]))
        # println(ds[s]'*y_best[s])
        obj_second_stage_best[s] = (ds[s]'*y_best[s])[1]
    end

    result["x"] = x_best
    result["y"] = y_best
    result["obj_first_stage_best"] = (x_best'*c)[1]
    result["obj_second_stage_best"] = obj_second_stage_best

    return result

end