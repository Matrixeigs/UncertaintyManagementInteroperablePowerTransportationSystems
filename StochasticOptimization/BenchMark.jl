using SparseArrays
include("/Solvers/MixedIntegerLinearProgrammingJUMP")

function result = two_stage_so_centralized(model_first_stage, model_second_stage, options)
# A centralize optimzation problem as a benchmark
    ns = length(model_second_stage)
    ps = model_first_stage.ps

    c = model_first_stage.c
    A = model_first_stage.A
    b = model_first_stage.b
    vtype_first_stage = model_first_stage.vtype
    lx = model_first_stage.lb
    ux = model_first_stage.ub
    NX = length(lx)

    Ts = cell(ns, 1)
    Ws = cell(ns, 1)
    hs = cell(ns, 1)
    ds = cell(ns, 1)
    ly = cell(ns, 1)
    uy = cell(ns, 1)
    con_obj = zeros(ns, 1)
    vtype_second_stage = cell(ns, 1)
    for s=1:ns
        Ts(s,1) = model_second_stage(s,1).T
        Ws(s,1) = model_second_stage(s,1).W
        hs(s,1) = model_second_stage(s,1).hs
        try()
            ds(s,1) = model_second_stage(s,1).ds
        catch()
            ds(s,1) = model_second_stage(s,1).d
        end
        ly(s,1) = model_second_stage(s,1).lb
        uy(s,1) = model_second_stage(s,1).ub
        if size(model_second_stage(s,1).vtype,1) .== 1
            model_second_stage(s,1).vtype = model_second_stage(s,1).vtype'
        end
        vtype_second_stage(s,1) = model_second_stage(s,1).vtype
    end

    NY = length(ds[1])
    if size(ps,1)==1
        ps=ps'
    end
    model_master.obj = [c ps]'
    model_master.lb = [lx zeros(ns,1)]'
    model_master.ub = [ux inf*ones(ns,1)]'
    model_master.A = [A, spzeros(size(A,1), ns)]
    model_master.rhs = b
    model_master.sense = model_first_stage.sense
    model_master.vtype = [vtype_first_stage repeat("C", ns, 1)]

    model_centralized = model_master

    for s = 1:ns
        model_centralized.obj = [model_centralized.obj zeros(NY,1)]'
        model_centralized.lb = [model_centralized.lb ly(s,1)]'
        model_centralized.ub = [model_centralized.ub uy(s,1)]'
        model_centralized.vtype = [model_centralized.vtype vtype_second_stage(s,1)]'
        model_centralized.A = [model_centralized.A, spzeros(size(model_centralized.A,1),NY) Ts(s,1), spzeros(size(Ts(s,1),1), ns + NY*(s-1)), Ws(s,1)]
        model_centralized.rhs = [model_centralized.rhs hs(s,1)]'
        model_centralized.sense = [model_centralized.sense repeat(">',length(hs(s,1)),1)]"
        temp = zeros(NX + ns + (s-1)*NY, 1);#
        temp[NX+s] = 1
        model_centralized.A = [model_centralized.A temp", -ds(s,1)']"
        model_centralized.rhs = [model_centralized.rhs 0]'
        model_centralized.sense = [model_centralized.sense ">']"
    #     result = gurobi[model_centralized,options]
    end
    result = gurobi[model_centralized,options]

    x_best = result.x[1:NX]
    y_best = cell(ns,1)
    obj_second_stage_best = zeros(ns,1)
    for s=1:ns
        y_best(s,1) = result.x[NX + ns + (s-1)*NY +  1: NX + ns + s*NY]
        obj_second_stage_best[s,1] = ds(s,1)'*y_best(s,1) + con_obj[s,1]
    end

    result.x = x_best
    result.y = y_best
    result.obj_first_stage_best = x_best'*c
    result.obj_second_stage_best = obj_second_stage_best

end