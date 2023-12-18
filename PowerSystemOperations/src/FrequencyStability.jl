# Run power flow to derive the basic setting points
# We export the result as multi-layer dictionary and need to 

using PowerModels
using Ipopt
using CSV
using DataFrames

pwd = pkgdir(PowerModels) # cases data location
result = solve_ac_opf(pwd*"/test/data/matpower/case24.m", Ipopt.Optimizer)

# df = DataFrame(result)
# CSV.write("result.csv", df)

solution = result["solution"]

function save_nested_dict_to_csv(dict::Dict, prefix::String = "")
    for (key, value) in dict
        if isa(value, Dict)
            nested_prefix = isempty(prefix) ? key : "$prefix-$key"
            save_nested_dict_to_csv(value, nested_prefix)
        else
            flat_dict = Dict("$prefix-$key" => value)
            df = DataFrame(flat_dict, copycols=false)
            CSV.write("$prefix.csv", df, append=!isempty(prefix))
        end
    end
end

save_nested_dict_to_csv(solution,"result")