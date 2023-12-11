#=
    Two-stage SIP optimization packages. The standard format is listed as follows:
    \min_{x_{b}, x_{c} \in X} 

    It should be noted that, the continuous part and integer part should be splited, as JuMP currently do not support the direct definition of variables using Strings.

=#

module TwoStageSIP

greet() = print("Hello World!")

end # module TrafficAssignment


