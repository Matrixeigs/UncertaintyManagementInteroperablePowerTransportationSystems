"""
Two-stage robust unit commitment for jointed wind hydro dispatch
@author: Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""
from pypower import loadcase, ext2int, makeBdc
from scipy.sparse import csr_matrix as sparse
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, concatenate, array, diag, eye
from Solvers.mixed_integer_programming_gurobi import mixed_integer_linear_programming as lp
from RobustOptimization.two_stage_robust_optimization import TwoStageRobustOptimization
import pandas as pd


def problem_formulation(PWMAX=10, PLMAX = 50, BETA=0.1, BETA_LOAD=0.0):
    """
    :param case: The test case for unit commitment problem for virtual power plant
    :return:
    """
    T = 24
    # The disturbance range of wind farm
    Price_energy = array([0.206716207449378,0.236805616543221,0.261173875249793,0.233147705859313,0.247100766764755,
                          0.219853978976507,0.243482127983914,0.261920984444443,0.270045782657531,0.286847887505983,
                          0.275403225203148,0.292543557821135,0.309320716045345,0.312282891483119,0.329495227435751,
                          0.291263217006437,0.283675266059621,0.271433683132372,0.248716542437820,0.242704942711648,
                          0.272605417316470,0.288923998792458,0.268587848457621,0.242311171642742]).reshape((T,1))
    Price_energy = Price_energy * 1000
    # Formulate the external power systems
    nw = 1
    nb = 1
    ng = 2
    ## Profiles
    # Wind profile
    WIND_PROFILE = array(
        [591.35, 714.50, 1074.49, 505.06, 692.78, 881.88, 858.48, 609.11, 559.95, 426.86, 394.54, 164.47, 27.15, 4.47,
         54.08, 109.90, 111.50, 130.44, 111.59, 162.38, 188.16, 216.98, 102.94, 229.53]).reshape((T, 1))
    WIND_PROFILE = WIND_PROFILE / WIND_PROFILE.max()
    WIND_PROFILE_FORECAST = zeros((T * nw, 1))
    Delta_wind = zeros((T * nw, 1))
    for i in range(T):
        WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw, :] = WIND_PROFILE[i] * PWMAX
        Delta_wind[i * nw:(i + 1) * nw, :] = WIND_PROFILE[i] * PWMAX * BETA

    # Load profile
    LOAD_PROFILE = array([0.632596195634005, 0.598783973523217, 0.580981513054525, 0.574328051348912, 0.584214221241601,
                          0.631074282084712, 0.708620833751212, 0.797665730618795, 0.877125330124026, 0.926981579915087,
                          0.947428654208872, 0.921588439808779, 0.884707317888543, 0.877717046100358, 0.880387289807107,
                          0.892056129442049, 0.909233443653261, 0.926748403704075, 0.968646575067696, 0.999358974358974,
                          0.979169591816267, 0.913517534182463, 0.806453715775750, 0.699930632166617]).reshape((T, 1))
    LOAD_FORECAST = zeros((T * nb, 1))
    Delta_load = zeros((T * nb, 1))
    for t in range(T):
        LOAD_FORECAST[t] = PLMAX * LOAD_PROFILE[t]
        Delta_load[t] = PLMAX * BETA_LOAD
    IL = Price_energy.max()/2
    il = Price_energy.max()*0.6

    # Generator information
    PMAX = array([10,10]).reshape((ng, 1))
    PMIN = array([0, 0]).reshape((ng, 1))
    START_UP_COST = array([100, 100]).reshape((ng, 1))
    SHUT_DOWN_COST = array([100, 100]).reshape((ng, 1))
    bg = array([200, 150]).reshape((ng, 1))
    ag = array([240, 260]).reshape((ng, 1))
    UP_LIMIT = [0] * ng
    DOWN_LIMIT = [0] * ng
    UP_DURATION = [0] * ng
    DOWN_DURATION = [0] * ng
    MIN_UP = 1
    MIN_DOWN = 1
    u0 = [1] * ng  # The initial generation status
    for g in range(ng):
        UP_LIMIT[g] = T - MIN_UP
        DOWN_LIMIT[g] = T - MIN_DOWN
        UP_DURATION[g] = 1
        DOWN_DURATION[g] = 0

    # bidding price information
    PDA_MAX = 100
    PDA_MIN = 0

    # Energy storage information
    ESS_MIN = 1
    ESS_MAX = 4
    PESS_CH_MAX = 2
    PESS_DC_MAX = 2
    EFF_DC = 0.95
    EFF_CH = 0.9
    E0 = 2
    c_ess = 100
    cVOLL = 1000
    PIM_MAX = 1e4
    # Define the first stage decision variables
    ON = 0
    OFF = ng
    IG = OFF + ng
    PDA = IG + ng
    PL = PDA + 1
    nx = PL + 1
    NX = nx * T
    lb = zeros((NX, 1))
    ub = zeros((NX, 1))
    c = zeros((NX, 1))
    vtypes = ["c"] * NX
    for t in range(T):
        for g in range(ng):
            # lower boundary information
            lb[t * nx + ON + g] = 0
            lb[t * nx + OFF + g] = 0
            lb[t * nx + IG + g] = 0
            # upper boundary information
            ub[t * nx + ON + g] = 1
            ub[t * nx + OFF + g] = 1
            ub[t * nx + IG + g] = 1
            # price information
            c[t * nx + ON + g] = START_UP_COST[g]
            c[t * nx + OFF + g] = SHUT_DOWN_COST[g]
            c[t * nx + IG + g] = bg[g]
            # variables types
            vtypes[t * nx + ON + g] = "b"
            vtypes[t * nx + OFF + g] = "b"
            vtypes[t * nx + IG + g] = "b"
            if t==0:
                lb[t * nx + IG + g] = u0[g]
                ub[t * nx + IG + g] = u0[g]
        lb[t * nx + PDA] = PDA_MIN
        lb[t * nx + PL] = 0
        ub[t * nx + PDA] = PDA_MAX
        ub[t * nx + PL] = LOAD_FORECAST[t]
        c[t * nx + PDA] = Price_energy[t]
        c[t * nx + PL] = IL

    # 2) Constraint set
    # 2.1) Status transformation of each unit
    Aeq = zeros((T * ng, NX))
    beq= zeros((T * ng, 1))
    for t in range(T):
        for g in range(ng):
            Aeq[t * ng + g, t * nx + ON + g] = -1
            Aeq[t * ng + g, t * nx + OFF + g] = 1
            Aeq[t * ng + g, t * nx + IG + g] = 1
            if t != 0:
                Aeq[t * ng + g, (t-1) * nx + IG + g] = -1
            else:
                beq[t * T + g] = 0

    # 2.2) Start up and shut down time limitation
    # 2.2.1) Up limit
    Aineq = zeros((sum(UP_LIMIT), NX))
    bineq = zeros((sum(UP_LIMIT), 1))
    for g in range(ng):
        for t in range(int(MIN_UP), T):
            for j in range(t - int(MIN_UP), t):
                Aineq[sum(UP_LIMIT[0:g]).astype(int) + t - int(MIN_UP), j * nx + ON + g] = 1
            Aineq[sum(UP_LIMIT[0:g]).astype(int) + t - int(MIN_UP), t * nx + IG + g] = -1
    # 2.2.2) Down limit
    Aineq_temp = zeros((sum(DOWN_LIMIT), NX))
    bineq_temp = ones((sum(DOWN_LIMIT), 1))
    for g in range(ng):
        for t in range(MIN_DOWN, T):
            for j in range(t - MIN_DOWN, t):
                Aineq_temp[sum(DOWN_LIMIT[0:g]).astype(int) + t - MIN_DOWN, j * nx + OFF + g] = 1
            Aineq_temp[sum(DOWN_LIMIT[0:g]).astype(int) + t - MIN_DOWN, t * nx + IG + g] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    model_first_stage = {"c": c,
                         "lb": lb,
                         "ub": ub,
                         "A": Aineq,
                         "b": bineq,
                         "Aeq": Aeq,
                         "beq": beq,
                         "vtypes": vtypes}
    # (xx, obj, success) = lp(model_first_stage["c"], Aeq=model_first_stage["Aeq"], beq=model_first_stage["beq"],
    #                         A=model_first_stage["A"],
    #                         b=model_first_stage["b"], xmin=model_first_stage["lb"], xmax=model_first_stage["ub"],
    #                         vtypes=model_first_stage["vtypes"], objsense="min")
    # xx = array(xx).reshape((len(xx), 1))

    ## Formualte the second stage decision making problem
    Pg = 0
    Pl = Pg + 1
    Pd = Pl + 1
    Pc = Pd + 1
    Eess = Pc + 1
    Plc_p = Eess + 1
    Plc_n = Plc_p + 1
    ny = Plc_n + 1
    NY = ny*T
    # Generate the lower and boundary for the first stage decision variables
    lb = zeros((NY, 1))
    ub = zeros((NY, 1))
    c = zeros((NY, 1))
    vtypes = ["c"] * NY
    nuw = 0
    nub = nuw + nw
    nu = nub + nb
    NU = nu*T
    u_mean = zeros((NU,1))
    u_delta = zeros((NU,1))
    for t in range(T):
        u_mean[t * nu + nuw] = WIND_PROFILE_FORECAST[t]
        u_mean[t * nu + nub] = LOAD_FORECAST[t]
        u_delta[t * nu + nuw] = Delta_wind[t]
        u_delta[t * nu + nub] = Delta_load[t]

    for t in range(T):
        # Pg information
        for g in range(ng):
            # lower boundary information
            lb[t * ny + Pg + g] = 0
            # upper boundary information
            ub[t * ny + Pg + g] = PMAX[g]
            # objective value
            c[t * ny + Pg + g] = ag[g]
        # Pl information
        lb[t * ny + Pl] = 0
        ub[t * ny + Pl] = PLMAX
        c[t * ny + Pl] = il
        # Pd information
        lb[t * ny + Pd] = 0
        ub[t * ny + Pd] = PESS_DC_MAX
        c[t * ny + Pd] = c_ess
        # Pc information
        lb[t * ny + Pc] = 0
        ub[t * ny + Pc] = PESS_CH_MAX
        c[t * ny + Pc] = c_ess
        # Eess information
        lb[t * ny + Eess] = ESS_MIN
        ub[t * ny + Eess] = ESS_MAX
        if t == T-1:
            lb[t * ny + Eess] = E0
            ub[t * ny + Eess] = E0
        # Plc_p information
        lb[t * ny + Plc_p] = 0
        ub[t * ny + Plc_p] = PIM_MAX
        c[t * ny + Plc_p] = cVOLL
        # Plc_n information
        lb[t * ny + Plc_n] = 0
        ub[t * ny + Plc_n] = PIM_MAX
        c[t * ny + Plc_n] = cVOLL

    # Generate correlate constraints
    # 3.1) Power balance constraints
    E = zeros((T, NX))
    M = zeros((T, NU))
    G = zeros((T, NY))
    h = zeros((T, 1))
    for t in range(T):
        # For the hydro units
        for g in range(ng):
            G[t, t*ny + Pg + g] = 1
        # For the wind farms
        for j in range(nw):
            M[t, t*nu + nuw + j] = 1
        # For the loads
        for j in range(nb):
            M[t, t*nu + nub + j] = -1

        G[t, t * ny + Pd] = 1
        G[t, t * ny + Pc] = -1
        G[t, t * ny + Pl] = 1
        G[t, t * ny + Plc_p] = 1
        G[t, t * ny + Plc_n] = -1
        E[t, t*nx + PDA] = 1

    # Update G,M,E,h
    G = concatenate([G, -G])
    M = concatenate([M, -M])
    E = concatenate([E, -E])
    h = concatenate([h, -h])

    # 3.2) Power range limitation
    #
    E_temp = zeros((T * ng, NX))
    M_temp = zeros((T * ng, NU))
    G_temp = zeros((T * ng, NY))
    h_temp = zeros((T * ng, 1))
    for t in range(T):
        for g in range(ng):
            G_temp[t * ng + g, t*ny + Pg + g] = 1
            E_temp[t * ng + g, t*ny + IG + g] = -PMIN[g]
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros((T * ng, NX))
    M_temp = zeros((T * ng, NU))
    G_temp = zeros((T * ng, NY))
    h_temp = zeros((T * ng, 1))
    for t in range(T):
        for g in range(ng):
            G_temp[t * ng + g, t*ny + Pg + g] = -1
            E_temp[t * ng + g, t*ny + IG + g] = PMAX[g]
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    # 3.3) Ramp rate constraints
    E_temp = zeros(((T-1) * ng, NX))
    M_temp = zeros(((T-1) * ng, NU))
    G_temp = zeros(((T-1) * ng, NY))
    h_temp = zeros(((T-1)* ng, 1))
    for t in range(T-1):
        for g in range(ng):
            G_temp[t * ng + g, t * ny + Pg + g] = 1
            G_temp[t * ng + g, (t+1) * ny + Pg + g] = -1
            E_temp[t * ng + g, t * ny + IG + g] = PMAX[g]
            E_temp[t * ng + g, (t+1) * ny + ON + g] = PMAX[g]
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros(((T-1) * ng, NX))
    M_temp = zeros(((T-1) * ng, NU))
    G_temp = zeros(((T-1) * ng, NY))
    h_temp = zeros(((T-1) * ng, 1))
    for t in range(T-1):
        for g in range(ng):
            G_temp[t * ng + g, t * ny + Pg + g] = -1
            G_temp[t * ng + g, (t+1) * ny + Pg + g] = 1
            E_temp[t * ng + g, t * ny + IG + g] = PMAX[g]
            E_temp[t * ng + g, (t+1) * ny + OFF + g] = PMAX[g]
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    # 3.4) Energy storage constraints
    E_temp = zeros((T, NX))
    M_temp = zeros((T, NU))
    G_temp = zeros((T, NY))
    h_temp = zeros((T, 1))
    for t in range(T):
        G_temp[t, t * ny + Eess] = 1
        G_temp[t, t * ny + Pc] = -EFF_CH
        G_temp[t, t * ny + Pd] = 1/EFF_DC
        if t == 0:
            h_temp[t, 0] = E0
        else:
            G_temp[t, (t - 1) * ny + Eess] = -1
    G = concatenate([G, G_temp, -G_temp])
    M = concatenate([M, M_temp, -M_temp])
    E = concatenate([E, E_temp, -E_temp])
    h = concatenate([h, h_temp, -h_temp])
    # 3.5) Load shedding range
    E_temp = zeros((T * nb, NX))
    M_temp = zeros((T * nb, NU))
    G_temp = zeros((T * nb, NY))
    h_temp = zeros((T * nb, 1))
    for t in range(T):
        for j in range(nb):
            G_temp[t * nb + j, t*ny + Pl + j] = -1
            E_temp[t * nb + j, t*nx + PL + j] = 1
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    # 3.9) Upper boundary and lower boundary information
    E_temp = zeros((NY, NX))
    M_temp = zeros((NY, NU))
    G_temp = eye(NY)
    h_temp = lb
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros((NY, NX))
    M_temp = zeros((NY, NU))
    G_temp = -eye(NY)
    h_temp = -ub
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    d = c
    # Test the second stage problem
    # u = u_mean - u_delta
    # model_second_stage = {"c": d,
    #                       "A": -G,
    #                       "b": M.dot(u) + E.dot(xx) - h}
    # (yy, obj_second_stage, success_second_stage) = lp(model_second_stage["c"],
    #                                                   A=model_second_stage["A"],
    #                                                   b=model_second_stage["b"], objsense="min")
    # yy = array(yy).reshape((len(yy), 1))
    #
    # (yy_dual, obj_dual, success_dual) = lp(model_second_stage["b"],
    #                                        Aeq=model_second_stage["A"].transpose(),
    #                                        beq=model_second_stage["c"],
    #                                        xmin=zeros((model_second_stage["A"].shape[0], 1)),
    #                                        objsense="max")

    # For every first stage solution, there exists a feabile solution for the second stage optimization.
    two_stage_robust_optimization = TwoStageRobustOptimization()
    (xx,uu,obj) = two_stage_robust_optimization.main(model_first_stage["c"], Aeq=model_first_stage["Aeq"],
                                            beq=model_first_stage["beq"], A=model_first_stage["A"],
                                            b=model_first_stage["b"],
                                            lb=model_first_stage["lb"], ub=model_first_stage["ub"],
                                            vtypes=model_first_stage["vtypes"], d=d, G=G, E=E, M=M, h=h, u_mean=u_mean,
                                            u_delta=u_delta, budget=array([[u_delta.shape[0]]]))

    # Decompose the first stage decision varialbes

    data = {"Price":Price_energy}
    writer = pd.ExcelWriter("data.xlsx")
    keys = list(data.keys())

    for i in range(len(keys)):
        df = pd.DataFrame(data[keys[i]])
        df.to_excel(writer, sheet_name=keys[i])
    writer.close()

    On = zeros((T, ng))
    Off = zeros((T, ng))
    Ig = zeros((T, ng))
    Pda = zeros((T, 1))
    Pl_da = zeros((T, 1))

    for t in range(T):
        for g in range(ng):
            On[t,g] = xx[t*nx+ON+g]
            Off[t,g] = xx[t*nx+OFF+g]
            Ig[t,g] = xx[t*nx+IG+g]

        Pda[t] = xx[t*nx+PDA]
        Pl_da[t] = xx[t * nx + PL]


    sol = {"START_UP": On,
           "SHUT_DOWN": Off,
           "IG": Ig,
           "PDA": Pda,
           "PL": Pl_da,
           "OBJ":[obj]}

    writer = pd.ExcelWriter("day_head_results.xlsx")
    keys = list(sol.keys())

    for i in range(len(keys)):
        df = pd.DataFrame(sol[keys[i]])
        df.to_excel(writer, sheet_name=keys[i])
    writer.close()


    # Derive the worst scenario
    u = u_mean - u_delta + 2*uu*u_delta
    model_second_stage = {"c": d,
                          "A": -G,
                          "b": M.dot(u) + E.dot(xx) - h}
    (yy, obj_second_stage, success_second_stage) = lp(model_second_stage["c"],
                                                      A=model_second_stage["A"],
                                                      b=model_second_stage["b"], objsense="min")
    pd_worst = u[nub:NU:nu]
    pw_worst = u[nuw:NU:nu]
    uncertain = {
        "PW_exp": WIND_PROFILE_FORECAST,
        "PW_max": WIND_PROFILE_FORECAST+Delta_wind,
        "PW_min": WIND_PROFILE_FORECAST-Delta_wind,
        "PW_worst": pw_worst,
        "Pd_exp": LOAD_FORECAST,
        "Pd_max": LOAD_FORECAST+Delta_load,
        "Pd_min": LOAD_FORECAST-Delta_load,
        "pd_worst":pd_worst}

    writer = pd.ExcelWriter("uncertainties.xlsx")
    keys = list(uncertain.keys())

    for i in range(len(keys)):
        df = pd.DataFrame(uncertain[keys[i]])
        df.to_excel(writer, sheet_name=keys[i])
    writer.close()

    # Obtain the best reaction
    pg_sec = zeros((T,ng))
    for t in range(T):
        for g in range(ng):
            pg_sec[t , g] = yy[t*ny + Pg + g]
    pl = yy[Pl:NY:ny]
    pdc = yy[Pd:NY:ny]
    pch = yy[Pc:NY:ny]
    eess = yy[Eess:NY:ny]

    sol_sec = {"Pg": pg_sec,
           "Pl": pl,
           "Pd": pdc,
           "Pc": pch,
           "EESS": eess}

    writer = pd.ExcelWriter("second_stage.xlsx")
    keys = list(sol_sec.keys())

    for i in range(len(keys)):
        df = pd.DataFrame(sol_sec[keys[i]])
        df.to_excel(writer, sheet_name=keys[i])
    writer.close()


    return sol


if __name__ == "__main__":
    model = problem_formulation(PWMAX=10, PLMAX=50, BETA=0.1, BETA_LOAD=0.1)

