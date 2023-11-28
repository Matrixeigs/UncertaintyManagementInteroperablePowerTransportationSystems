"""
Unit commitment problem for hybrid AC/DC MGs
@author: Zhao Tianyang
@e-mail: matrixeigs@gmail.com

Supporting documents:
Resilient Energy Management for Hybrid AC/DC MGs

"""

from idx_format_hybrid_AC_DC import PBIC_A2D, PBIC_D2A, PESS_CH0, PESS_DC0, PG0, PPV0, PUG, \
    NX_MG, QBIC, QG0, QUG, NESS, NRES, EESS0, PAC, PDC

from numpy import zeros, ones, concatenate, eye, tile
from scipy.sparse import vstack, lil_matrix

from random import random
class UnitCommitment():
    """
    Unit commitment problem for hybrid AC/DC MGs for stochastic optimization
    """

    def __init__(self, mg):
        self.T = len(mg["PD"]["AC"])
        self.ng = len(mg["DG"])
        self.NX = NX_MG
        self.ness = NESS
        self.nres = NRES
        self.nv_uncertainty = self.T * 2 + self.T * NRES

    def stochastic_optimization(self, mg, ns=4):
        model_first_stage = self.first_stage_problem_formulation(mg=mg)
        # Formulate the second stage scenarios

        us = zeros((ns, self.nv_uncertainty))
        ws = ones(ns) / ns
        for i in range(ns):
            us[i, 0: 2 * self.T] = concatenate([mg["PD"]["AC"], mg["PD"]["DC"]])
            for j in range(self.nres):
                us[i, (2 + j) * self.T:(3 + j) * self.T] = concatenate([mg["PV"][j]["PROFILE"]])

        u_mean = us[0, :]
        u_delta = us[0, :] * 0.1
        for i in range(ns):
            for j in range(self.nv_uncertainty):
                us[i, j] += (2 * random() - 1) * u_delta[j]

        model_second_stage = [0] * ns
        for i in range(ns):
            model_second_stage[i] = self.second_stage_problem_formulation(mg=mg, u=us[i, :])
        
        return model_first_stage, model_second_stage

    def first_stage_problem_formulation(self, mg):
        ng = self.ng
        self.ALPHA = 0
        self.BETA = 1
        self.IG = 2
        self.PG = 3
        self.RG = 4
        self.FUEL = 5

        T = self.T
        _nv_first_stage = ng * 6
        nv_first_stage = _nv_first_stage * T

        self._nv_first_stage = _nv_first_stage
        self.nv_first_stage = nv_first_stage
        # Obtain the initial status, start-up and shut down of generators
        Ig0 = zeros(ng)
        MIN_DOWN = zeros(ng)
        MIN_UP = zeros(ng)
        for i in range(ng):
            Ig0[i] = mg["DG"][i]["I0"]
            MIN_DOWN[i] = mg["DG"][i]["MU"]
            MIN_UP[i] = mg["DG"][i]["MD"]
        Ig0 = Ig0.astype(int)
        MIN_DOWN = MIN_DOWN.astype(int)
        MIN_UP = MIN_UP.astype(int)

        # The decision variables includes the start-up, shut down, generator output, reserve capacity, fuel refilling plan
        alpha_l = zeros(ng)
        beta_l = zeros(ng)
        Ig_l = zeros(ng)
        pg_l = zeros(ng)  # Boundary for DGs within distribution networks
        rg_l = zeros(ng)
        fuel_l = zeros(ng)

        alpha_u = ones(ng)
        beta_u = ones(ng)
        Ig_u = ones(ng)
        pg_u = zeros(ng)
        rg_u = zeros(ng)
        fuel_u = zeros(ng)

        c_alpha = zeros(ng)
        c_beta = zeros(ng)
        c_ig = zeros(ng)
        cg = zeros(ng)
        cr = zeros(ng)
        cfuel = zeros(ng)

        for i in range(ng):
            pg_u[i] = mg["DG"][i]["PMAX"]
            rg_u[i] = mg["DG"][i]["PMAX"]
            fuel_u[i] = mg["DG"][i]["TANK"] - mg["DG"][i]["TANK0"]

            c_ig[i] = mg["DG"][i]["FUEL"] * mg["DG"][i]["PMIN"] * mg["DG"][i]["FUEL_PRICE"]
            cg[i] = mg["DG"][i]["FUEL"] * mg["DG"][i]["FUEL_PRICE"]
            cfuel[i] = mg["DG"][i]["FUEL_PRICE"]

        # Formulate the boundaries
        lb = concatenate([tile(concatenate([alpha_l, beta_l, Ig_l, pg_l, rg_l, fuel_l]), T)])
        ub = concatenate([tile(concatenate([alpha_u, beta_u, Ig_u, pg_u, rg_u, fuel_u]), T)])
        # Objective value
        c = concatenate([tile(concatenate([c_alpha, c_beta, c_ig, cg, cr, cfuel]), T)])
        # Variable types
        vtypes = (["b"] * ng * 3 + ["c"] * ng * 3) * T
        ## Constraint sets
        # 1) Pg+Rg<=PguIg
        A = lil_matrix((ng * T, nv_first_stage))
        b = zeros(ng * T)
        for t in range(T):
            for j in range(ng):
                A[t * ng + j, t * _nv_first_stage + ng * self.PG + j] = 1
                A[t * ng + j, t * _nv_first_stage + ng * self.RG + j] = 1
                A[t * ng + j, t * _nv_first_stage + ng * self.IG + j] = -pg_u[j]
        # 2) Pg-Rg>=IgPgl
        A_temp = lil_matrix((ng * T, nv_first_stage))
        b_temp = zeros(ng * T)
        for t in range(T):
            for j in range(ng):
                A_temp[t * ng + j, t * _nv_first_stage + ng * self.PG + j] = -1
                A_temp[t * ng + j, t * _nv_first_stage + ng * self.RG + j] = 1
                A_temp[t * ng + j, t * _nv_first_stage + ng * self.IG + j] = pg_l[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        # 3) Start-up and shut-down constraints of DGs
        UP_LIMIT = zeros(ng).astype(int)
        DOWN_LIMIT = zeros(ng).astype(int)
        for i in range(ng):
            UP_LIMIT[i] = T - MIN_UP[i]
            DOWN_LIMIT[i] = T - MIN_DOWN[i]
        # 3.1) Up limit
        A_temp = lil_matrix((sum(UP_LIMIT), nv_first_stage))
        b_temp = zeros(sum(UP_LIMIT))
        for i in range(ng):
            for t in range(MIN_UP[i], T):
                for k in range(t - MIN_UP[i], t):
                    A_temp[sum(UP_LIMIT[0:i]) + t - MIN_UP[i], k * _nv_first_stage + ng * self.ALPHA + i] = 1
                A_temp[sum(UP_LIMIT[0:i]) + t - MIN_UP[i], t * _nv_first_stage + ng * self.IG + i] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # # 3.2) Down limit
        A_temp = lil_matrix((sum(DOWN_LIMIT), nv_first_stage))
        b_temp = ones(sum(DOWN_LIMIT))
        for i in range(ng):
            for t in range(MIN_DOWN[i], T):
                for k in range(t - MIN_DOWN[i], t):
                    A_temp[sum(DOWN_LIMIT[0:i]) + t - MIN_DOWN[i], k * _nv_first_stage + ng * self.BETA + i] = 1
                A_temp[sum(DOWN_LIMIT[0:i]) + t - MIN_DOWN[i], t * _nv_first_stage + ng * self.IG + i] = 1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 4) Status transformation of each unit
        Aeq = lil_matrix((T * ng, nv_first_stage))
        beq = zeros(T * ng)
        for i in range(ng):
            for t in range(T):
                Aeq[i * T + t, t * _nv_first_stage + ng * self.ALPHA + i] = 1
                Aeq[i * T + t, t * _nv_first_stage + ng * self.BETA + i] = -1
                Aeq[i * T + t, t * _nv_first_stage + ng * self.IG + i] = -1
                if t != 0:
                    Aeq[i * T + t, (t - 1) * _nv_first_stage + ng * self.IG + i] = 1
                else:
                    beq[i * T + t] = -Ig0[i]

        model_first_stage = {"c": c.reshape((nv_first_stage, 1)),
                             "lb": lb.reshape((nv_first_stage, 1)),
                             "ub": ub.reshape((nv_first_stage, 1)),
                             "vtypes": vtypes,
                             "A": A.tolil(),
                             "b": b.reshape((len(b), 1)),
                             "Aeq": Aeq.tolil(),
                             "beq": beq.reshape((len(beq), 1)), }

        # (sol_first_stage, obj_first_stage, success) = milp(model_first_stage["c"], Aeq=model_first_stage["Aeq"],
        #                                                    beq=model_first_stage["beq"],
        #                                                    A=model_first_stage["A"], b=model_first_stage["b"],
        #                                                    vtypes=model_first_stage["vtypes"],
        #                                                    xmax=model_first_stage["ub"], xmin=model_first_stage["lb"])
        # sol = self.first_stage_solution_validation(sol_first_stage)
        return model_first_stage

    def second_stage_problem_formulation(self, mg, u):
        """
        Second-stage problem formulation for hybrid AC/DC MGs
        :param mg:
        :return: model of second stage problem, i.e., real-time scheduling of hybrid AC/DC microgrids
        """

        T = self.T
        ng = self.ng
        ness = self.ness
        nres = self.nres
        nv_uncertainty = self.nv_uncertainty
        nv_first_stage = self.nv_first_stage
        self._nv_second_stage = NX_MG
        ## 1) boundary information and objective function
        _nv_second_stage = NX_MG
        nv_second_stage = NX_MG * T
        self.nv_second_stage = nv_second_stage

        lb = zeros(nv_second_stage)
        ub = zeros(nv_second_stage)
        c = zeros(nv_second_stage)
        q = zeros(nv_second_stage)
        vtypes = ["c"] * nv_second_stage
        for t in range(T):
            ## 1.1) lower boundary
            for i in range(ng):
                lb[t * NX_MG + PG0 + i] = 0
                lb[t * NX_MG + QG0 + i] = mg["DG"][i]["QMIN"]
            lb[t * NX_MG + PUG] = 0
            lb[t * NX_MG + QUG] = mg["UG"]["QMIN"]
            lb[t * NX_MG + PBIC_D2A] = 0
            lb[t * NX_MG + PBIC_A2D] = 0
            lb[t * NX_MG + QBIC] = -mg["BIC"]["SMAX"]
            for i in range(ness):
                lb[t * NX_MG + PESS_CH0 + i] = 0
                lb[t * NX_MG + PESS_DC0 + i] = 0
                lb[t * NX_MG + EESS0 + i] = mg["ESS"][i]["EMIN"]
            for i in range(nres):
                lb[t * NX_MG + PPV0 + i] = 0
            lb[t * NX_MG + PAC] = 0
            lb[t * NX_MG + PDC] = 0

            ## 1.2) upper boundary
            for i in range(ng):
                ub[t * NX_MG + PG0 + i] = mg["DG"][i]["PMAX"]
                ub[t * NX_MG + QG0 + i] = mg["DG"][i]["QMAX"]
            ub[t * NX_MG + PUG] = mg["UG"]["PMAX"]
            ub[t * NX_MG + QUG] = mg["UG"]["QMAX"]
            ub[t * NX_MG + PBIC_D2A] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + PBIC_A2D] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + QBIC] = mg["BIC"]["SMAX"]
            for i in range(ness):
                ub[t * NX_MG + PESS_CH0 + i] = mg["ESS"][i]["PCH_MAX"]
                ub[t * NX_MG + PESS_DC0 + i] = mg["ESS"][i]["PDC_MAX"]
                ub[t * NX_MG + EESS0 + i] = mg["ESS"][i]["EMAX"]
            for i in range(nres):
                ub[t * NX_MG + PPV0 + i] = max(mg["PV"][i]["PROFILE"])
            ub[t * NX_MG + PAC] = mg["PD"]["AC_MAX"]
            ub[t * NX_MG + PDC] = mg["PD"]["DC_MAX"]

            ## 1.3) Objective functions
            for i in range(ng):
                c[t * NX_MG + PG0 + i] = mg["DG"][i]["FUEL_PRICE"] * mg["DG"][i]["FUEL"]
            for i in range(ness):
                c[t * NX_MG + PESS_CH0 + i] = mg["ESS"][i]["COST"]
                c[t * NX_MG + PESS_DC0 + i] = mg["ESS"][i]["COST"]
            for i in range(nres):
                c[t * NX_MG + PPV0 + i] = mg["PV"][i]["COST"]

            c[t * NX_MG + PUG] = mg["UG"]["COST"][t]
            c[t * NX_MG + PAC] = mg["VOLL"]
            c[t * NX_MG + PDC] = mg["VOLL"]

            ## 1.4) Upper and lower boundary information
            if t == T - 1:
                for i in range(ness):
                    lb[t * NX_MG + EESS0 + i] = mg["ESS"][i]["E0"]
                    ub[t * NX_MG + EESS0 + i] = mg["ESS"][i]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = lil_matrix((T, nv_second_stage))
        beq = zeros(T)
        Eeq = lil_matrix((T, nv_uncertainty))
        Teq = lil_matrix((T, nv_first_stage))
        for t in range(T):
            for i in range(ng):
                Aeq[t, t * NX_MG + PG0 + i] = 1
            Aeq[t, t * NX_MG + PUG] = 1
            Aeq[t, t * NX_MG + PBIC_A2D] = -1
            Aeq[t, t * NX_MG + PBIC_D2A] = mg["BIC"]["EFF_D2A"]
            Aeq[t, t * NX_MG + PAC] = 1
            Eeq[t, t] = -1
        # b) DC bus equation
        Aeq_temp = lil_matrix((T, nv_second_stage))
        beq_temp = zeros(T)
        Eeq_temp = lil_matrix((T, nv_uncertainty))
        Teq_temp = lil_matrix((T, nv_first_stage))

        for t in range(T):
            Aeq_temp[t, t * NX_MG + PBIC_A2D] = mg["BIC"]["EFF_A2D"]
            Aeq_temp[t, t * NX_MG + PBIC_D2A] = -1
            for i in range(ness):
                Aeq_temp[t, t * NX_MG + PESS_CH0 + i] = -1
                Aeq_temp[t, t * NX_MG + PESS_DC0 + i] = 1
            for i in range(nres):
                Aeq_temp[t, t * NX_MG + PPV0 + i] = 1
            Aeq_temp[t, t * NX_MG + PDC] = 1
            Eeq_temp[t, T + t] = -1

        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        Eeq = vstack([Eeq, Eeq_temp])
        Teq = vstack([Teq, Teq_temp])

        # c) AC reactive power balance equation
        Aeq_temp = lil_matrix((T, nv_second_stage))
        beq_temp = zeros(T)
        Eeq_temp = lil_matrix((T, nv_uncertainty))
        Teq_temp = lil_matrix((T, nv_first_stage))

        for t in range(T):
            Aeq_temp[t, t * NX_MG + QUG] = 1
            Aeq_temp[t, t * NX_MG + PAC] = mg["QD"]["AC"][t] / mg["PD"]["AC"][t]
            Aeq_temp[t, t * NX_MG + QBIC] = 1
            for i in range(ng):
                Aeq_temp[t, t * NX_MG + QG0 + i] = 1
            beq_temp[t] = mg["QD"]["AC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        Eeq = vstack([Eeq, Eeq_temp])
        Teq = vstack([Teq, Teq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = lil_matrix((T * ness, nv_second_stage))
        beq_temp = zeros(T)
        Eeq_temp = lil_matrix((T * ness, nv_uncertainty))
        Teq_temp = lil_matrix((T * ness, nv_first_stage))

        for t in range(T):
            for i in range(ness):
                Aeq_temp[t + i * T, t * NX_MG + EESS0 + i] = 1
                Aeq_temp[t + i * T, t * NX_MG + PESS_CH0 + i] = -mg["ESS"][i]["EFF_CH"]
                Aeq_temp[t + i * T, t * NX_MG + PESS_DC0 + i] = 1 / mg["ESS"][i]["EFF_DC"]
                if t == 0:
                    beq_temp[t + i * T] = mg["ESS"][i]["E0"]
                else:
                    Aeq_temp[t + i * T, (t - 1) * NX_MG + EESS0 + i] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        Eeq = vstack([Eeq, Eeq_temp])
        Teq = vstack([Teq, Teq_temp])

        # 3) Formualte inequality constraints
        # 3.1) Ppv <= Ppvmax
        A = lil_matrix((T * nres, nv_second_stage))
        b = zeros(T * nres)
        E = lil_matrix((T * nres, nv_uncertainty))
        Tx = lil_matrix((T * nres, nv_first_stage))
        for t in range(T):
            for i in range(nres):
                A[t + i * T, t * NX_MG + PPV0 + i] = 1
                E[t + i * T, T * 2 + i * T + t] = -1
        # 3.2) Pac <= Pd_AC
        A_temp = lil_matrix((T, nv_second_stage))
        b_temp = zeros(T)
        E_temp = lil_matrix((T, nv_uncertainty))
        T_temp = lil_matrix((T, nv_first_stage))

        for t in range(T):
            A_temp[t, t * NX_MG + PAC] = 1
            E_temp[t, t] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        E = vstack([E, E_temp])
        Tx = vstack([Tx, T_temp])

        # 3.3) Pdc <= Pd_DC
        A_temp = lil_matrix((T, nv_second_stage))
        b_temp = zeros(T)
        E_temp = lil_matrix((T, nv_uncertainty))
        T_temp = lil_matrix((T, nv_first_stage))

        for t in range(T):
            A_temp[t, t * NX_MG + PDC] = 1
            E_temp[t, T + t] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        E = vstack([E, E_temp])
        Tx = vstack([Tx, T_temp])

        # The coupling constriants between the first stage and second stage variables
        ## IV) Formulate the coupling constraints between the first-stage and second-stage problems
        # 1) -Pg -Rg + pg <= 0
        _nv_first_stage = self._nv_first_stage
        nv_first_stage = _nv_first_stage * T
        Ts = lil_matrix((ng * T, nv_first_stage))
        Ws = lil_matrix((ng * T, nv_second_stage))
        hs = zeros(ng * T)
        E_temp = lil_matrix((ng * T, nv_uncertainty))

        for t in range(T):
            for j in range(ng):
                Ts[t * ng + j, t * _nv_first_stage + ng * self.PG + j] = -1
                Ts[t * ng + j, t * _nv_first_stage + ng * self.RG + j] = -1
                Ws[t * ng + j, t * _nv_second_stage + PG0 + j] = 1
        E = vstack([E, E_temp])

        # 2) Pg-Rg - pg <= 0
        Ts_temp = lil_matrix((ng * T, nv_first_stage))
        Ws_temp = lil_matrix((ng * T, nv_second_stage))
        hs_temp = zeros(ng * T)
        E_temp = lil_matrix((ng * T, nv_uncertainty))

        for t in range(T):
            for j in range(ng):
                Ts_temp[t * ng + j, t * _nv_first_stage + ng * self.PG + j] = 1
                Ts_temp[t * ng + j, t * _nv_first_stage + ng * self.RG + j] = -1
                Ws_temp[t * ng + j, t * _nv_second_stage + PG0 + j] = -1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        E = vstack([E, E_temp])

        # 3) Qg <= IgQg_max
        Ts_temp = lil_matrix((ng * T, nv_first_stage))
        Ws_temp = lil_matrix((ng * T, nv_second_stage))
        hs_temp = zeros(ng * T)
        E_temp = lil_matrix((ng * T, nv_uncertainty))

        for t in range(T):
            for j in range(ng):
                Ts_temp[t * ng + j, t * _nv_first_stage + ng * self.IG + j] = -mg["DG"][j]["QMAX"]
                Ws_temp[t * ng + j, t * _nv_second_stage + QG0 + j] = 1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        E = vstack([E, E_temp])

        # 4) Qg >= IgQg_min
        Ts_temp = lil_matrix((ng * T, nv_first_stage))
        Ws_temp = lil_matrix((ng * T, nv_second_stage))
        hs_temp = zeros(ng * T)
        E_temp = lil_matrix((ng * T, nv_uncertainty))

        for t in range(T):
            for j in range(ng):
                Ts_temp[t * ng + j, t * _nv_first_stage + ng * self.IG + j] = mg["DG"][j]["QMIN"]
                Ws_temp[t * ng + j, t * _nv_second_stage + QG0 + j] = -1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        E = vstack([E, E_temp])

        # T = vstack([Tx, Ts])  # First stage
        # W = vstack([A, Ws])  # Second stage
        # h = concatenate([b, hs])

        T = vstack([Tx, Ts, Teq, -Teq, zeros((2 * nv_second_stage, nv_first_stage))])  # First stage
        E = vstack([E, Eeq, -Eeq, zeros((2 * nv_second_stage, nv_uncertainty))])  # Uncertainty factors
        W = vstack([A, Ws, Aeq, -Aeq, eye(nv_second_stage), -eye(nv_second_stage)])  # Second stage
        h = concatenate([b, hs, beq, -beq, ub, -lb])

        model_micro_grid = {"c": c.reshape((nv_second_stage, 1)),
                            "q": q,
                            "lb": lb.reshape((nv_second_stage, 1)),
                            "ub": ub.reshape((nv_second_stage, 1)),
                            "vtypes": vtypes,
                            "A": A.tolil(),
                            "b": b.reshape((len(b), 1)),
                            "E": T.tolil(),
                            "G": W.tolil(),
                            "M": E.tolil(),
                            "h": h.reshape((len(h), 1)),
                            "Eeq": None,
                            "Geq": None,
                            "Meq": None,
                            "heq": None
                            }

        return model_micro_grid

    def first_stage_solution_validation(self, sol):
        """
        Validation of the first-stage solution
        :param sol: The first stage solution
        :return: the first stage solution
        """
        T = self.T
        ng = self.ng
        # Set-points of DGs within DSs, MGs and ESSs
        _nv_first_stage = self._nv_first_stage
        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        Ig = zeros((ng, T))
        Pg = zeros((ng, T))
        Rg = zeros((ng, T))
        Fuel = zeros((ng, T))

        for i in range(T):
            alpha[:, i] = sol[_nv_first_stage * i:_nv_first_stage * i + ng]
            beta[:, i] = sol[_nv_first_stage * i + ng:_nv_first_stage * i + ng * 2]
            Ig[:, i] = sol[_nv_first_stage * i + ng * 2:_nv_first_stage * i + ng * 3]
            Pg[:, i] = sol[_nv_first_stage * i + ng * 3:_nv_first_stage * i + ng * 4]
            Rg[:, i] = sol[_nv_first_stage * i + ng * 4:_nv_first_stage * i + ng * 5]
            Fuel[:, i] = sol[_nv_first_stage * i + ng * 5:_nv_first_stage * i + ng * 6]

        sol_first_stage = {"alpha": alpha,
                           "beta": beta,
                           "ig": Ig,
                           "rg": Rg,
                           "pg": Pg,
                           "fuel": Fuel,
                           }
        return sol_first_stage

    def second_stage_solution_validation(self, sol):
        T = self.T
        ng = self.ng
        ness = self.ness
        nres = self.nres
        _nv_second_stage = self._nv_second_stage
        mg_sol = {}
        mg_sol["pg"] = zeros((ng, T))
        mg_sol["qg"] = zeros((ng, T))
        mg_sol["pug"] = zeros((1, T))
        mg_sol["qug"] = zeros((1, T))
        mg_sol["pbic_a2d"] = zeros((1, T))
        mg_sol["pbic_d2a"] = zeros((1, T))
        mg_sol["qbic"] = zeros((1, T))
        mg_sol["pess_ch"] = zeros((ness, T))
        mg_sol["pess_dc"] = zeros((ness, T))
        mg_sol["eess"] = zeros((ness, T))
        mg_sol["ppv"] = zeros((nres, T))
        mg_sol["pac"] = zeros((1, T))
        mg_sol["pdc"] = zeros((1, T))

        for t in range(T):
            for i in range(ng):
                mg_sol["pg"][i, t] = sol[t * _nv_second_stage + PG0 + i]
                mg_sol["qg"][i, t] = sol[t * _nv_second_stage + QG0 + i]
            mg_sol["pug"][0, t] = sol[t * _nv_second_stage + PUG]
            mg_sol["qug"][0, t] = sol[t * _nv_second_stage + QUG]
            mg_sol["pbic_a2d"][0, t] = sol[t * _nv_second_stage + PBIC_A2D]
            mg_sol["pbic_d2a"][0, t] = sol[t * _nv_second_stage + PBIC_D2A]
            mg_sol["qbic"][0, t] = sol[t * _nv_second_stage + QBIC]
            for i in range(ness):
                mg_sol["pess_ch"][i, t] = sol[t * _nv_second_stage + PESS_CH0 + i]
                mg_sol["pess_dc"][i, t] = sol[t * _nv_second_stage + PESS_DC0 + i]
                mg_sol["eess"][i, t] = sol[t * _nv_second_stage + EESS0 + i]
            for i in range(nres):
                mg_sol["ppv"][i, t] = sol[t * _nv_second_stage + PPV0 + i]
            mg_sol["pac"][0, t] = sol[t * _nv_second_stage + PAC]
            mg_sol["pdc"][0, t] = sol[t * _nv_second_stage + PDC]

        return mg_sol


if __name__ == "__main__":

    from cases_unit_commitment import micro_grid

    mg = micro_grid
    mg["VOLL"] = 1e9
    unit_commitment = UnitCommitment(mg=mg)
    (model_first_stage, model_second_stage) = unit_commitment.stochastic_optimization(mg=mg)
    print(model_first_stage)
