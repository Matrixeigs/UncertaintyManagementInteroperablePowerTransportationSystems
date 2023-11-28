"""
Two stage unit commitment with N-k constraints
This work is to detect the failure case considering the failures of generators and transmission lines

The data is to test the IEEE-24 test system
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, array, eye, random
from scipy.sparse import csr_matrix as sparse
from pypower.idx_brch import F_BUS, T_BUS, BR_X, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, BUS_I
from pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN, RAMP_AGC, RAMP_10, RAMP_30
from pypower.idx_cost import STARTUP
from TestCasesUnitCommitment.data_format_contigency import ALPHA, BETA, IG, PG, RS, RU, RD, THETA, PL, NG
from Solvers.mixed_integer_programming_gurobi import mixed_integer_linear_programming as milp
from RobustOptimization.two_stage_robust_optimization_uncertainty_set import TwoStageRobustOptimization
from utils.save_files import save_to_excel
import os


class TwoStageStochasticUnitCommitment():

    def __init__(self):
        self.name = "Two-stage stochastic unit commitment"

    def problem_formulation_first_stage(self, case, delta=0.50, delta_r=0.03):
        baseMVA, bus, gen, branch, gencost, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "gencost"], case["Load_profile"]
        MIN_UP = -3

        # Modify the bus, gen and branch matrix, the adjustment of
        bus[:, BUS_I] = bus[:, BUS_I] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1
        gen[:, RAMP_10] = gencost[:, -8] * 20
        gen[:, RAMP_AGC] = gencost[:, -8] * 10
        gen[:, RAMP_30] = gencost[:, -8] * 60
        pd_index = case['bus'][case['bus'][:, PD] > 0, BUS_I]
        Pd = case['bus'][case['bus'][:, PD] > 0, PD]

        ng = shape(case['gen'])[0]  # number of schedule injections
        nl = shape(case['branch'])[0]  ## number of branches
        nb = shape(case['bus'])[0]  ## number of branches
        nd = len(pd_index)
        T = case["Load_profile"].shape[0]

        # Pass the information
        self.ng = ng
        self.nb = nb
        self.nd = nd
        self.nl = nl
        self.bus = bus
        self.branch = branch
        self.gen = gen
        self.T = T
        self.Pd = Pd
        self.Profile = profile
        self.nu = ng + nl

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = r_[range(nl), range(nl)]  ## double set of row indices

        ## connection matrix
        Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))), (nb, ng))
        Cd = sparse((ones(nd), (pd_index, arange(nd))), (nb, nd))

        u0 = [0] * ng  # The initial generation status
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        UP_DURATION = [0] * ng
        DOWN_DURATION = [0] * ng

        for g in range(ng):
            u0[g] = int(gencost[g, 9] > 0)
            UP_LIMIT[g] = T - int(gencost[g, MIN_UP])
            DOWN_LIMIT[g] = T - int(gencost[g, MIN_UP])
            UP_DURATION[g] = int(max(0, gencost[g, 9]))
            DOWN_DURATION[g] = int(-min(0, gencost[g, 9]))

        nx = NG * T * ng + nb * T + nl * T + T  # generations, bus, branch and operational reserve
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        vtypes = ["c"] * nx
        self.nx = nx

        for t in range(T):
            for g in range(ng):
                # lower boundary
                lb[ALPHA * ng * T + t * ng + g] = 0
                lb[BETA * ng * T + t * ng + g] = 0
                if t < max(UP_DURATION[g] - gencost[g, MIN_UP], 0):
                    lb[IG * ng * T + t * ng + g] = 1
                else:
                    lb[IG * ng * T + t * ng + g] = 0
                lb[PG * ng * T + t * ng + g] = 0
                lb[RS * ng * T + t * ng + g] = 0
                lb[RU * ng * T + t * ng + g] = 0
                lb[RD * ng * T + t * ng + g] = 0

                # upper boundary
                ub[ALPHA * ng * T + t * ng + g] = 1
                ub[BETA * ng * T + t * ng + g] = 1
                if t < max(DOWN_DURATION[g] - gencost[g, MIN_UP], 0):
                    ub[IG * ng * T + t * ng + g] = 0
                else:
                    ub[IG * ng * T + t * ng + g] = 1
                ub[PG * ng * T + t * ng + g] = gen[g, PMAX]
                ub[RS * ng * T + t * ng + g] = gen[g, RAMP_10]
                ub[RU * ng * T + t * ng + g] = gen[g, RAMP_AGC]
                ub[RD * ng * T + t * ng + g] = gen[g, RAMP_AGC]
                # variable types
                vtypes[IG * ng * T + t * ng + g] = "B"
            # Operational reserve
            lb[NG * T * ng + nb * T + nl * T + t] = 0
            ub[NG * T * ng + nb * T + nl * T + t] = max(gen[:, PMAX].tolist())

        # The bus angle
        for t in range(T):
            for i in range(nb):
                lb[NG * ng * T + t * nb + i] = -360
                ub[NG * ng * T + t * nb + i] = 360
                if bus[i, BUS_TYPE] == REF:
                    lb[NG * ng * T + t * nb + i] = 0
                    ub[NG * ng * T + t * nb + i] = 0

        # The power flow
        for t in range(T):
            for k in range(nl):
                lb[NG * ng * T + T * nb + t * nl + k] = -branch[k, RATE_A] * 1
                ub[NG * ng * T + T * nb + t * nl + k] = branch[k, RATE_A] * 1

        # 2) Constraint set
        c = zeros((nx, 1))
        for t in range(T):
            for g in range(ng):
                # cost, the linear objective value
                c[ALPHA * ng * T + t * ng + g] = gencost[g, STARTUP]
                c[IG * ng * T + t * ng + g] = gencost[g, 6]
                c[PG * ng * T + t * ng + g] = gencost[g, 5]
                # reserve cost
                c[RS * ng * T + t * ng + g] = gencost[g, 5] / 10
                c[RU * ng * T + t * ng + g] = gencost[g, 5] / 10
                c[RD * ng * T + t * ng + g] = gencost[g, 5] / 10

        # 2) Constraint set
        # 2.1) Power balance equation, for each node
        Aeq = zeros((T * nb, nx))
        beq = zeros((T * nb, 1))
        for t in range(T):
            # For the unit
            Aeq[t * nb:(t + 1) * nb, PG * ng * T + t * ng:PG * ng * T + (t + 1) * ng] = Cg.todense()
            # For the transmission lines
            Aeq[t * nb:(t + 1) * nb, NG * ng * T + T * nb + t * nl: NG * ng * T + T * nb + (t + 1) * nl] = \
                -(Cft.transpose()).todense()

            beq[t * nb:(t + 1) * nb, 0] = Cd * profile[t] * Pd

        self.Cg = Cg
        self.Cd = Cd
        self.Cft = Cft

        # 2.2) Status transformation of each unit
        Aeq_temp = zeros((T * ng, nx))
        beq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aeq_temp[t * ng + g, ALPHA * ng * T + t * ng + g] = -1
                Aeq_temp[t * ng + g, BETA * ng * T + t * ng + g] = 1
                Aeq_temp[t * ng + g, IG * ng * T + t * ng + g] = 1
                if i != 0:
                    Aeq_temp[t * ng + g, IG * ng * T + (t - 1) * ng + g] = -1
                else:
                    beq_temp[t * T + g, 0] = 0

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)

        # 2.3) Transmission line flows
        Aeq_temp = zeros((T * nl, nx))
        beq_temp = zeros((T * nl, 1))
        X = zeros((nl, nl))
        for k in range(nl):
            X[k, k] = 1 / branch[k, BR_X]

        for t in range(T):
            # For the unit
            Aeq_temp[t * nl:(t + 1) * nl, NG * ng * T + T * nb + t * nl:NG * ng * T + T * nb + (t + 1) * nl] = -eye(nl)
            Aeq_temp[t * nl:(t + 1) * nl, NG * ng * T + t * nb: NG * ng * T + (t + 1) * nb] = X.dot(Cft.todense())

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)
        # 2.4) Start-up and shut-down commands limitation
        Aineq = zeros((T * ng, nx))
        bineq = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq[t * ng + g, ALPHA * ng * T + t * ng + g] = 1
                Aineq[t * ng + g, BETA * ng * T + t * ng + g] = 1
                bineq[t * ng + g, 0] = 1

        # 2.4) Power range limitation
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t * ng + g, IG * ng * T + t * ng + g] = gen[g, PMIN]
                Aineq_temp[t * ng + g, PG * ng * T + t * ng + g] = -1
                Aineq_temp[t * ng + g, RD * ng * T + t * ng + g] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t * ng + g, IG * ng * T + t * ng + g] = -gen[g, PMAX]
                Aineq_temp[t * ng + g, PG * ng * T + t * ng + g] = 1
                Aineq_temp[t * ng + g, RU * ng * T + t * ng + g] = 1
                Aineq_temp[t * ng + g, RS * ng * T + t * ng + g] = 1

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Start up and shut down time limitation
        # 2.5.1) Up limit
        Aineq_temp = zeros((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for g in range(ng):
            for t in range(int(gencost[g, MIN_UP]), T):
                for j in range(t - int(gencost[g, MIN_UP]), t):
                    Aineq_temp[sum(UP_LIMIT[0:g]) + t - int(gencost[g, MIN_UP]), ALPHA * ng * T + j * ng + g] = 1
                Aineq_temp[sum(UP_LIMIT[0:g]) + t - int(gencost[g, MIN_UP]), IG * ng * T + t * ng + g] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Down limit
        Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
        bineq_temp = ones((sum(DOWN_LIMIT), 1))
        for g in range(ng):
            for t in range(int(gencost[g, MIN_UP]), T):
                for j in range(t - int(gencost[g, MIN_UP]), t):
                    Aineq_temp[sum(DOWN_LIMIT[0:g]) + t - int(gencost[g, MIN_UP]), BETA * ng * T + j * ng + g] = 1
                Aineq_temp[sum(DOWN_LIMIT[0:g]) + t - int(gencost[g, MIN_UP]), IG * ng * T + t * ng + g] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.6) Ramp constraints:
        # 2.6.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for g in range(ng):
            for t in range(T - 1):
                Aineq_temp[g * (T - 1) + t, PG * ng * T + (t + 1) * ng + g] = 1
                Aineq_temp[g * (T - 1) + t, PG * ng * T + t * ng + g] = -1
                Aineq_temp[g * (T - 1) + t, ALPHA * ng * T + (t + 1) * ng + g] = gen[g, PMIN]  # Start-up capacity
                bineq_temp[g * (T - 1) + t] = gen[g, RAMP_30]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # # 2.6.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for g in range(ng):
            for t in range(T - 1):
                Aineq_temp[g * (T - 1) + t, PG * ng * T + (t + 1) * ng + g] = -1
                Aineq_temp[g * (T - 1) + t, PG * ng * T + t * ng + g] = 1
                Aineq_temp[g * (T - 1) + t, BETA * ng * T + (t + 1) * ng + g] = gen[g, PMIN]  # Shut-down capacity
                bineq_temp[g * (T - 1) + t] = gen[g, RAMP_30]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.7)  Reserve constraints
        # 2.7.1) Rs<=Ig*RAMP_10
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t * ng + g, IG * ng * T + t * ng + g] = -gen[g, RAMP_10]
                Aineq_temp[t * ng + g, RS * ng * T + t * ng + g] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.7.2) ru<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t * ng + g, IG * ng * T + t * ng + g] = -gen[g, RAMP_AGC]
                Aineq_temp[t * ng + g, RU * ng * T + t * ng + g] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.7.3) rd<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t * ng + g, IG * ng * T + t * ng + g] = -gen[g, RAMP_AGC]
                Aineq_temp[t * ng + g, RD * ng * T + t * ng + g] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.8)  Up and down reserve for the forecasting errors
        # 2.8.1) Spinning reserve limitation
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t * ng + g, RS * ng * T + t * ng + g] = 1
                Aineq_temp[t * ng + g, PG * ng * T + t * ng + g] = 1
                Aineq_temp[t * ng + g, NG * T * ng + nb * T + nl * T + t] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t, RS * ng * T + t * ng + g] = -1
            Aineq_temp[t, NG * T * ng + nb * T + nl * T + t] = delta
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.8.2) Up regulation reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t, RU * ng * T + t * ng + g] = -1
            bineq_temp[t] -= delta_r * profile[t] * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.8.3) Down regulation reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for t in range(T):
            for g in range(ng):
                Aineq_temp[t, RD * ng * T + t * ng + g] = -1
            bineq_temp[t] -= delta_r * profile[t] * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {"c": c,
                 "lb": lb,
                 "ub": ub,
                 "A": Aineq,
                 "b": bineq,
                 "Aeq": Aeq,
                 "beq": beq,
                 "vtypes": vtypes}

        (xx, obj, success) = milp(model["c"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"], objsense="min")

        sol = self.result_check(xx)
        save_to_excel(sol, "TwoStageRO", os.getcwd())

        return model

    def uncertainty_set_formulation(self, kg=2, kl=2):
        ng = self.ng
        nl = self.nl
        nu = ng + nl
        self.nu = nu
        u_mean = 0.5 * ones((nu, 1))
        u_delta = 0.5 * ones((nu, 1))
        Cu = concatenate([-ones((1, ng)), zeros((1, nl))], axis=1)
        fu = (kg - ng) * ones((1, 1))
        Cu_temp = concatenate([zeros((1, ng)), -ones((1, nl))], axis=1)
        fu_temp = (kl - nl) * ones((1, 1))
        Cu = concatenate([Cu, Cu_temp], axis=0)
        fu = concatenate([fu, fu_temp], axis=0)

        model = {"Cu": Cu,
                 "fu": fu,
                 "u_mean": u_mean,
                 "u_delta": u_delta}

        return model

    def problem_formulation_second_stage(self):
        """
        Problem formulation for the second stage problem
        Found problem: The upper and lower boundary should be modified to the standard format
        :return:
        """

        ng = self.ng
        nb = self.nb
        nd = self.nd
        nl = self.nl
        T = self.T
        bus = self.bus
        branch = self.branch
        gen = self.gen
        Pd = self.Pd
        profile = self.Profile
        # Number of variables
        nx = ng * T + nd * T + nb * T + nl * T + ng * T + ng * T
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        c = zeros((nx, 1))
        pg = 0
        pd = 1
        VOLL = 4 * 10 ** 3

        for t in range(T):
            for g in range(ng):
                # real-time power dispatch
                lb[pg * ng * T + t * ng + g] = 0
                ub[pg * ng * T + t * ng + g] = gen[g, PMAX]
                # I*(Pg+Rs)
                lb[ng * T + nd * T + nb * T + nl * T + t * ng + g] = 0
                ub[ng * T + nd * T + nb * T + nl * T + t * ng + g] = gen[g, PMAX]
                # I*(Pg-Rs)
                lb[ng * T + nd * T + nb * T + nl * T + ng * T + t * ng + g] = -gen[g, PMAX]
                ub[ng * T + nd * T + nb * T + nl * T + ng * T + t * ng + g] = gen[g, PMAX]
            for j in range(nd):
                # load shedding at different buses
                lb[pd * ng * T + t * nd + j] = 0
                ub[pd * ng * T + t * nd + j] = Pd[j] * profile[t]
                c[pd * ng * T + t * nd + j] = VOLL

        for t in range(T):
            for i in range(nb):
                # The bus angle
                lb[ng * T + nd * T + t * nb + i] = -360
                ub[ng * T + nd * T + t * nb + i] = 360
                if bus[i, BUS_TYPE] == REF:
                    lb[ng * T + nd * T + t * nb + i] = 0
                    ub[ng * T + nd * T + t * nb + i] = 0

        for t in range(T):
            for k in range(nl):
                # The power flow
                lb[ng * T + nd * T + nb * T + t * nl + k] = -branch[k, RATE_A]
                ub[ng * T + nd * T + nb * T + t * nl + k] = branch[k, RATE_A]
        # Construct the constraint set
        # 3.1) Power balance constraints
        NX = self.nx
        nu = self.nu
        Cg = self.Cg
        Cd = self.Cd
        Cft = self.Cft

        E_temp = zeros((T * nb, NX))
        M_temp = zeros((T * nb, nu))
        G_temp = zeros((T * nb, nx))
        h_temp = zeros((T * nb, 1))
        for t in range(T):
            # For the unit
            G_temp[t * nb:(t + 1) * nb, pg * ng * T + t * ng:pg * ng * T + (t + 1) * ng] = Cg.todense()
            # For the load shedding
            G_temp[t * nb:(t + 1) * nb, pd * ng * T + t * nd:pd * ng * T + (t + 1) * nd] = Cd.todense()
            # For the transmission lines
            G_temp[t * nb:(t + 1) * nb, ng * T + nd * T + nb * T + t * nl: ng * T + nd * T + nb * T + (t + 1) * nl] = \
                -(Cft.transpose()).todense()
            h_temp[t * nb:(t + 1) * nb, 0] = Cd * Pd * profile[t]
        # Update G,M,E,h
        G = concatenate([G_temp, -G_temp])
        M = concatenate([M_temp, -M_temp])
        E = concatenate([E_temp, -E_temp])
        h = concatenate([h_temp, -h_temp])
        # 3.2 Line flow equation
        E_temp = zeros((T * nl, NX))
        M_temp = zeros((T * nl, nu))
        G_temp = zeros((T * nl, nx))
        h_temp = zeros((T * nl, 1))

        X = zeros((nl, nl))
        for i in range(nl):
            X[i, i] = 1 / branch[i, BR_X]

        for i in range(T):
            # For the unit
            G_temp[i * nl:(i + 1) * nl,
            ng * T + nd * T + nb * T + i * nl:ng * T + nd * T + nb * T + (i + 1) * nl] = -eye(nl)
            G_temp[i * nl:(i + 1) * nl, ng * T + nd * T + i * nb: ng * T + nd * T + (i + 1) * nb] = X.dot(Cft.todense())
            M_temp[i * nl:(i + 1) * nl, ng:] = -diag(branch[:, RATE_A])
            h_temp[i * nl:(i + 1) * nl, 0] = -branch[:, RATE_A]

        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        E_temp = zeros((T * nl, NX))
        M_temp = zeros((T * nl, nu))
        G_temp = zeros((T * nl, nx))
        h_temp = zeros((T * nl, 1))
        for i in range(T):
            # For the unit
            G_temp[i * nl:(i + 1) * nl,
            ng * T + nd * T + nb * T + i * nl:ng * T + nd * T + nb * T + (i + 1) * nl] = eye(nl)
            G_temp[i * nl:(i + 1) * nl, ng * T + nd * T + i * nb: ng * T + nd * T + (i + 1) * nb] = -X.dot(
                Cft.todense())
            M_temp[i * nl:(i + 1) * nl, ng:] = -diag(branch[:, RATE_A])
            h_temp[i * nl:(i + 1) * nl, 0] = -branch[:, RATE_A]
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        # 3.3) Power range limitation
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, pg * ng * T + t * ng + g] = -1
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + t * ng + g] = 1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, pg * ng * T + t * ng + g] = 1
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + ng * T + t * ng + g] = -1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 1.1
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + t * ng + g] = 1
                E_temp[t * ng + g, PG * ng * T + t * ng + g] = -1
                E_temp[t * ng + g, RS * ng * T + t * ng + g] = -1
                M_temp[t * ng + g, g] = -gen[g, PMAX]
                h_temp[t * ng + g, 0] = -gen[g, PMAX]
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 1.2
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + t * ng + g] = -1
                E_temp[t * ng + g, PG * ng * T + t * ng + g] = 1
                E_temp[t * ng + g, RS * ng * T + t * ng + g] = 1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 1.3
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + t * ng + g] = -1
                M_temp[t * ng + g, g] = gen[g, PMAX]
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 2.1
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + ng * T + t * ng + g] = 1
                E_temp[t * ng + g, PG * ng * T + t * ng + g] = -1
                E_temp[t * ng + g, RS * ng * T + t * ng + g] = 1
                M_temp[t * ng + g, g] = -gen[g, PMAX]
                h_temp[t * ng + g, 0] = -gen[g, PMAX]
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 2.2
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + ng * T + t * ng + g] = -1
                E_temp[t * ng + g, PG * ng * T + t * ng + g] = 1
                E_temp[t * ng + g, RS * ng * T + t * ng + g] = -1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 2.3
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for t in range(T):
            for g in range(ng):
                G_temp[t * ng + g, ng * T + nd * T + nb * T + nl * T + ng * T + t * ng + g] = -1
                M_temp[t * ng + g, g] = gen[g, PMAX]
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        # 3.5) Upper and lower boundary information
        E_temp = zeros((nx, NX))
        M_temp = zeros((nx, nu))
        G_temp = eye(nx)
        h_temp = lb
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        E_temp = zeros((nx, NX))
        M_temp = zeros((nx, nu))
        G_temp = -eye(nx)
        h_temp = -ub

        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        d = c

        model = {"G": G,
                 "M": M,
                 "E": E,
                 "h": h,
                 "d": d}
        # Modify the lower boundary

        return model

    def result_check(self, sol):
        """

        :param sol: The solution of mathematical
        :return:
        """
        T = self.T
        ng = self.ng
        nl = self.nl
        nb = self.nb

        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        ig = zeros((ng, T))
        pg = zeros((ng, T))
        Rs = zeros((ng, T))
        ru = zeros((ng, T))
        rd = zeros((ng, T))

        theta = zeros((nb, T))
        pf = zeros((nl, T))
        Qor = zeros((1, T))

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                Rs[j, i] = sol[RS * ng * T + i * ng + j]
                ru[j, i] = sol[RU * ng * T + i * ng + j]
                rd[j, i] = sol[RD * ng * T + i * ng + j]

        for i in range(T):
            for j in range(nb):
                theta[j, i] = sol[NG * ng * T + i * nb + j]

        for i in range(T):
            for j in range(nl):
                pf[j, i] = sol[NG * ng * T + T * nb + i * nl + j]

        for i in range(T):
            Qor[0, i] = sol[NG * ng * T + T * nb + T * nl + i]

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RS": Rs,
                    "RU": ru,
                    "RD": rd,
                    "THETA": theta,
                    "PF": pf,
                    "QOR": Qor}

        return solution

    def result_second_stage_check(self,sol):
        T = self.T
        ng = self.ng
        nl = self.nl
        nb = self.nb

        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        ig = zeros((ng, T))
        pg = zeros((ng, T))
        Rs = zeros((ng, T))
        ru = zeros((ng, T))
        rd = zeros((ng, T))

        theta = zeros((nb, T))
        pf = zeros((nl, T))
        Qor = zeros((1, T))

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                Rs[j, i] = sol[RS * ng * T + i * ng + j]
                ru[j, i] = sol[RU * ng * T + i * ng + j]
                rd[j, i] = sol[RD * ng * T + i * ng + j]

        for i in range(T):
            for j in range(nb):
                theta[j, i] = sol[NG * ng * T + i * nb + j]

        for i in range(T):
            for j in range(nl):
                pf[j, i] = sol[NG * ng * T + T * nb + i * nl + j]

        for i in range(T):
            Qor[0, i] = sol[NG * ng * T + T * nb + T * nl + i]

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RS": Rs,
                    "RU": ru,
                    "RD": rd,
                    "THETA": theta,
                    "PF": pf,
                    "QOR": Qor}

        return solution


if __name__ == "__main__":
    from TestCasesUnitCommitment.case24 import case24

    case_base = case24()
    profile = array(
        [1.75, 1.65, 1.58, 1.54, 1.55, 1.60, 1.73, 1.77, 1.86, 2.07, 2.29, 2.36, 2.42, 2.44, 2.49, 2.56, 2.56, 2.47,
         2.46, 2.37, 2.37, 2.33, 1.96, 1.96]) / 3
    case_base["Load_profile"] = profile

    two_stage_unit_commitment = TwoStageStochasticUnitCommitment()
    ro = TwoStageRobustOptimization()

    model_first_stage = two_stage_unit_commitment.problem_formulation_first_stage(case_base)
    model_second_stage = two_stage_unit_commitment.problem_formulation_second_stage()
    model_uncertainty = two_stage_unit_commitment.uncertainty_set_formulation(kg=2, kl=2)
    sol = ro.main(c=model_first_stage["c"], Aeq=model_first_stage["Aeq"], beq=model_first_stage["beq"],
                  A=model_first_stage["A"], b=model_first_stage["b"], lb=model_first_stage["lb"],
                  ub=model_first_stage["ub"], vtypes=model_first_stage["vtypes"], d=model_second_stage["d"],
                  G=model_second_stage["G"], E=model_second_stage["E"], M=model_second_stage["M"],
                  h=model_second_stage["h"], u_mean=model_uncertainty["u_mean"], u_delta=model_uncertainty["u_delta"],
                  Cu=model_uncertainty["Cu"], fu=model_uncertainty["fu"])

    sol = two_stage_unit_commitment.result_check(sol)

    save_to_excel(sol, "TwoStageRO_N_k", os.getcwd())

