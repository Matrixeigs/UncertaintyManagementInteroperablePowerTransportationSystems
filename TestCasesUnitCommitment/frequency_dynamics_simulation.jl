# Using PowerSimulationsDynamics.jl to simulate the frequency dynamics of a power system

using PowerSystems
using PowerSimulationsDynamics
using PowerSystemCaseBuilder
using Sundials
using Plots
omib_sys = build_system(PSIDSystems, "OMIB System")
time_span = (0.0, 30.0)
perturbation_trip = BranchTrip(1.0, Line, "BUS 1-BUS 2-i_1")
sim = Simulation!(ResidualModel, omib_sys, pwd(), time_span, perturbation_trip)
x0_init = read_initial_conditions(sim)
show_states_initial_value(sim)
small_sig = small_signal_analysis(sim)
summary_eigenvalues(small_sig)
execute!(sim, IDA(), dtmax = 0.02, saveat = 0.02, enable_progress_bar = false)
results = read_results(sim)
angle = get_state_series(results, ("generator-102-1", :δ));
plot(angle, xlabel = "time", ylabel = "rotor angle [rad]", label = "gen-102-1")

pwd()
