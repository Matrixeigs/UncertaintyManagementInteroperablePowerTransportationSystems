using JuMP, CPLEX

function add_annotation(
    model::JuMP.Model,
    variable_classification::Dict;
    all_variables::Bool = true,
)
    num_variables = sum(length(it) for it in values(variable_classification))
    if all_variables
        @assert num_variables == JuMP.num_variables(model)
    end
    indices, annotations = CPXINT[], CPXLONG[]
    for (key, value) in variable_classification
        for variable_ref in value
            push!(indices, variable_ref.index.value - 1)
            push!(annotations, CPX_BENDERS_MASTERVALUE + key)
        end
    end
    cplex = backend(model)
    index_p = Ref{CPXINT}()
    CPXnewlongannotation(
        cplex.env,
        cplex.lp,
        CPX_BENDERS_ANNOTATION,
        CPX_BENDERS_MASTERVALUE,
    )
    CPXgetlongannotationindex(
        cplex.env,
        cplex.lp,
        CPX_BENDERS_ANNOTATION,
        index_p,
    )
    CPXsetlongannotations(
        cplex.env,
        cplex.lp,
        index_p[],
        CPX_ANNOTATIONOBJ_COL,
        length(indices),
        indices,
        annotations,
    )
    return
end

# Problem

function illustrate_full_annotation()
    c_1, c_2 = [1, 4], [2, 3]
    dim_x, dim_y = length(c_1), length(c_2)
    b = [-2; -3]
    A_1, A_2 = [1 -3; -1 -3], [1 -2; -1 -1]
    model = JuMP.direct_model(CPLEX.Optimizer())
    set_optimizer_attribute(model, "CPXPARAM_Benders_Strategy", 1)
    @variable(model, x[1:dim_x] >= 0, Bin)
    @variable(model, y[1:dim_y] >= 0)
    variable_classification = Dict(0 => [x[1], x[2]], 1 => [y[1], y[2]])
    @constraint(model, A_2 * y + A_1 * x .<= b)
    @objective(model, Min, c_1' * x + c_2' * y)
    add_annotation(model, variable_classification)
    optimize!(model)
    x_optimal = value.(x)
    y_optimal = value.(y)
    println("x: $(x_optimal), y: $(y_optimal)")
end

function illustrate_partial_annotation()
    c_1, c_2 = [1, 4], [2, 3]
    dim_x, dim_y = length(c_1), length(c_2)
    b = [-2; -3]
    A_1, A_2 = [1 -3; -1 -3], [1 -2; -1 -1]
    model = JuMP.direct_model(CPLEX.Optimizer())
    # Note that the "CPXPARAM_Benders_Strategy" has to be set to 2 if partial
    # annotation is provided. If "CPXPARAM_Benders_Strategy" is set to 1, then
    # the following error will be thrown:
    # `CPLEX Error  2002: Invalid Benders decomposition.`
    set_optimizer_attribute(model, "CPXPARAM_Benders_Strategy", 2)
    @variable(model, x[1:dim_x] >= 0, Bin)
    @variable(model, y[1:dim_y] >= 0)
    variable_classification = Dict(0 => [x[1]], 1 => [y[1], y[2]])
    @constraint(model, A_2 * y + A_1 * x .<= b)
    @objective(model, Min, c_1' * x + c_2' * y)
    add_annotation(model, variable_classification; all_variables = false)
    optimize!(model)
    x_optimal = value.(x)
    y_optimal = value.(y)
    println("x: $(x_optimal), y: $(y_optimal)")
end