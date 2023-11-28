"""
Dynamic optimal power flow with multiple microgrids
"""

from distribution_system_optimization.test_cases import case33
from micro_grids.test_cases.cases_unit_commitment import micro_grid
from transportation_systems.test_cases import case3, TIME, LOCATION

from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack
from numpy import flatnonzero as find
from numpy import array, tile, arange

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, RATE_A
from pypower.idx_bus import PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from solvers.mixed_integer_quadratic_constrained_cplex import mixed_integer_quadratic_constrained_programming as miqcp
from copy import deepcopy

from distribution_system_optimization.data_format.idx_opf import PBIC_AC2DC, PG, PESS_DC, PBIC_DC2AC, PUG, PESS_CH, RUG, \
    RESS, RG, EESS, NX_MG, QBIC, QUG, QG


class DynamicOptimalPowerFlowTess():
    def __init__(self):
        self.name = "Dynamic optimal power flow with tess"

    def main(self, case, microgrids, profile, tess, traffic_networks):
        """
        Main entrance for network reconfiguration problems
        :param case: electric network information
        :param profile: load profile within the distribution networks
        :param micrgrids: dictionary for microgrids
        :param tess: dictionary for tess
        :return: network reconfiguration, distribution network status, and microgrid status
        """
        # Time spans
        T = len(profile)
        self.T = T
        # NUmber of microgrids
        nmg = len(microgrids)
        self.nmg = nmg
        # Number of tess
        nev = len(tess)
        self.nev = nev
        # Number of buses in the transportation networks
        nb_traffic = traffic_networks["bus"].shape[0]
        self.nb_traffic = nb_traffic

        # 1) Formulate the constraints for each system
        # 1.1) Distribution networks
        model_distribution_networks = self.problem_formualtion_distribution_networks(case=case, profile=profile,
                                                                                     micro_grids=microgrids, tess=tess,
                                                                                     traffic_networks=traffic_networks)
        # 1.2) Microgrids
        model_microgrids = {}
        for i in range(nmg):
            model_microgrids[i] = self.problem_formulation_microgrid(micro_grid=microgrids[i])
        # 1.3) Transportation energy storage systems
        model_tess = {}
        for i in range(nev):
            model_tess[i] = self.problem_formulation_tess(tess=tess[i], traffic_networks=traffic_networks)

        # 2) System level modelling
        # 2.1) Merge the model between distribution networks and microgrdis
        nVariables_distribution_network = len(model_distribution_networks["c"])
        if model_distribution_networks["Aeq"] is not None:
            neq_distribution_network = model_distribution_networks["Aeq"].shape[0]
        else:
            neq_distribution_network = 0
        if model_distribution_networks["A"] is not None:
            nineq_distribution_network = model_distribution_networks["A"].shape[0]
        else:
            nineq_distribution_network = 0

        nVariables = int(nVariables_distribution_network)
        neq = int(neq_distribution_network)
        nineq = int(nineq_distribution_network)

        nVariables_index = zeros(nmg + 1)
        neq_index = zeros(nmg + 1)
        nineq_index = zeros(nmg + 1)

        nVariables_index[0] = int(nVariables_distribution_network)
        neq_index[0] = int(neq_distribution_network)
        nineq_index[0] = int(nineq_distribution_network)
        for i in range(nmg):
            nVariables_index[i + 1] = nVariables_index[i] + len(model_microgrids[i]["c"])
            neq_index[i + 1] = neq_index[i] + model_microgrids[i]["Aeq"].shape[0]
            nineq_index[i + 1] = nineq_index[i] + model_microgrids[i]["A"].shape[0]
            nVariables += len(model_microgrids[i]["c"])
            neq += int(model_microgrids[i]["Aeq"].shape[0])
            nineq += int(model_microgrids[i]["A"].shape[0])

        lx = model_distribution_networks["lb"]
        ux = model_distribution_networks["ub"]
        c = model_distribution_networks["c"]
        vtypes = model_distribution_networks["vtypes"]

        if model_distribution_networks["beq"] is not None:
            beq = model_distribution_networks["beq"]
        else:
            beq = zeros(0)

        if model_distribution_networks["b"] is not None:
            b = model_distribution_networks["b"]
        else:
            b = zeros(0)

        Qc = model_distribution_networks["Qc"]
        q = model_distribution_networks["q"]

        A = zeros((int(nineq_index[-1]), int(nVariables_index[-1])))
        Aeq = zeros((int(neq_index[-1]), int(nVariables_index[-1])))

        if model_distribution_networks["Aeq"] is not None:
            Aeq[0:neq_distribution_network, 0:nVariables_distribution_network] = model_distribution_networks["Aeq"]
        if model_distribution_networks["A"] is not None:
            A[0:nineq_distribution_network, 0:nVariables_distribution_network] = model_distribution_networks["A"]

        for i in range(nmg):
            lx = concatenate([lx, model_microgrids[i]["lb"]])
            ux = concatenate([ux, model_microgrids[i]["ub"]])
            c = concatenate([c, model_microgrids[i]["c"]])
            q = concatenate([q, model_microgrids[i]["q"]])
            vtypes += model_microgrids[i]["vtypes"]
            beq = concatenate([beq, model_microgrids[i]["beq"]])
            b = concatenate([b, model_microgrids[i]["b"]])
            Aeq[int(neq_index[i]):int(neq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
                model_microgrids[i]["Aeq"]
            A[int(nineq_index[i]):int(nineq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
                model_microgrids[i]["A"]

        # Add coupling constraints, between the microgrids and distribution networks
        Ay2x = zeros((2 * nmg * T, int(nVariables_index[-1] - nVariables_index[0])))
        for i in range(T):
            for j in range(nmg):
                Ay2x[i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX_MG + PUG] = -1
                Ay2x[nmg * T + i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX_MG + QUG] = -1

        Aeq_temp = concatenate([model_distribution_networks["Ax2y"], Ay2x], axis=1)
        beq_temp = zeros(2 * nmg * T)

        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 2.2) Merge the model between distribution networks and transportation networks
        NX_traffic = self.NX_traffic

        nVariables_index_tess = zeros(nev + 1)
        neq_index_tess = zeros(nev + 1)
        nineq_index_tess = zeros(nev + 1)
        nVariables_index_tess[0] = nVariables_index[-1]
        neq_index_tess[0] = Aeq.shape[0]
        nineq_index_tess[0] = A.shape[0]

        for i in range(nev):
            nVariables_index_tess[i + 1] = nVariables_index_tess[i] + len(model_tess[i]["c"])
            neq_index_tess[i + 1] = neq_index_tess[i] + model_tess[i]["Aeq"].shape[0]
            nineq_index_tess[i + 1] = nineq_index_tess[i] + model_tess[i]["A"].shape[0]
            nVariables += len(model_tess[i]["c"])
            neq += int(model_tess[i]["Aeq"].shape[0])
            nineq += int(model_tess[i]["A"].shape[0])

            c = concatenate([c, model_tess[i]["c"]])
            q = concatenate([q, model_tess[i]["q"]])
            lx = concatenate([lx, model_tess[i]["lb"]])
            ux = concatenate([ux, model_tess[i]["ub"]])
            vtypes += model_tess[i]["vtypes"]
            beq = concatenate([beq, model_tess[i]["beq"]])
            b = concatenate([b, model_tess[i]["b"]])

        A_full = zeros((int(nineq_index_tess[-1]), int(nVariables_index_tess[-1])))
        Aeq_full = zeros((int(neq_index_tess[-1]), int(nVariables_index_tess[-1])))

        if Aeq is not None:
            Aeq_full[0:int(neq_index_tess[0]), 0:int(nVariables_index_tess[0])] = Aeq
        if A is not None:
            A_full[0:int(nineq_index_tess[0]), 0:int(nVariables_index_tess[0])] = A

        for i in range(nev):
            Aeq_full[int(neq_index_tess[i]):int(neq_index_tess[i + 1]),
            int(nVariables_index_tess[i]):int(nVariables_index_tess[i + 1])] = model_tess[i]["Aeq"]

            A_full[int(nineq_index_tess[i]):int(nineq_index_tess[i + 1]),
            int(nVariables_index_tess[i]):int(nVariables_index_tess[i + 1])] = model_tess[i]["A"]

        # Coupling constraints between distribution networks and tess
        Az2x = zeros((2 * nb_traffic * T, int(nVariables_index_tess[-1] - nVariables_index_tess[0])))
        n_stops = self.n_stops
        NX_status = self.nl_traffic

        for i in range(nev):
            Az2x[0:n_stops, i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + 2 * n_stops] = \
                -eye(n_stops)  # Discharging
            Az2x[0:n_stops, i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 3 * n_stops] = \
                eye(n_stops)  # Charging
            Az2x[n_stops:2 * n_stops, i * NX_traffic + NX_status + 3 * n_stops:i * NX_traffic + NX_status + 4 * n_stops] \
                = -eye(n_stops)  # Spinning reserve

        Aeq_temp = concatenate(
            [model_distribution_networks["Ax2z"], zeros((2 * n_stops, int(nVariables_index[-1] - nVariables_index[0]))),
             Az2x], axis=1)
        beq_temp = zeros(2 * nb_traffic * T)

        Aeq = concatenate([Aeq_full, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        A = A_full
        # 3) Solve the problem
        rc = zeros(len(Qc))
        (xx, obj, success) = miqcp(c, q, Aeq=Aeq, beq=beq, vtypes=vtypes, A=A, b=b, Qc=Qc, rc=rc, xmin=lx, xmax=ux)

        # 4) Check the solutions, including microgrids and distribution networks
        # 4.1) Scheduling plan of distribution networks
        sol_distribution_network = self.solution_check_distribution_network(xx[0:nVariables_distribution_network])
        # 4.2) Scheduling plan of each MG
        # a) Energy storage system group
        sol_microgrids = self.solution_check_microgrids(xx=xx, nVariables_index=nVariables_index)

        sol_tess = self.solution_check_tess(sol=xx[int(nVariables_index_tess[0]):int(nVariables_index_tess[-1])])

        return sol_distribution_network, sol_microgrids, sol_tess

    def problem_formualtion_distribution_networks(self, case, profile, micro_grids, tess, traffic_networks):
        T = self.T

        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nmg = self.nmg
        nev = self.nev
        nb_traffic = self.nb_traffic

        self.nl = nl
        self.nb = nb
        self.ng = ng

        m = zeros(nmg)  ## list of integration index
        Pmg_l = zeros(nmg)  ## list of lower boundary
        Pmg_u = zeros(nmg)  ## list of upper boundary
        Qmg_l = zeros(nmg)  ## list of lower boundary
        Qmg_u = zeros(nmg)  ## list of upper boundary
        for i in range(nmg):
            m[i] = micro_grids[i]["BUS"]
            Pmg_l[i] = micro_grids[i]["UG"]["PMIN"] / 1000 / baseMVA
            Pmg_u[i] = micro_grids[i]["UG"]["PMAX"] / 1000 / baseMVA
            Qmg_l[i] = micro_grids[i]["UG"]["QMIN"] / 1000 / baseMVA
            Qmg_u[i] = micro_grids[i]["UG"]["QMAX"] / 1000 / baseMVA

        n = traffic_networks["bus"][:, -1]  ## list of integration index
        Pev_l = zeros(nb_traffic)  ## lower boundary for energy exchange
        Pev_u = zeros(nb_traffic)  ## upper boundary for energy exchange
        Rev_l = zeros(nb_traffic)  ## lower boundary for spinning reserve
        Rev_u = zeros(nb_traffic)  ## upper boundary for spinning reserve
        for i in range(nb_traffic):
            for j in range(nev):
                Pev_l[i] = Pev_l[i] - tess[j]["PCMAX"] / 1000 / baseMVA
                Pev_u[i] = Pev_u[i] + tess[j]["PDMAX"] / 1000 / baseMVA
                Rev_l[i] = 0
                Rev_u[i] = Rev_u[i] + (tess[j]["PCMAX"] + tess[j]["PDMAX"]) / 1000 / baseMVA

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        self.f = f  ## record from bus for each branch

        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (m, range(nmg))), (nb, nmg))
        Cev = sparse((ones(nb_traffic), (n, range(nb_traffic))), (nb, nb_traffic))

        Branch_R = branch[:, BR_R]
        Branch_X = branch[:, BR_X]
        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        Slmax = branch[:, RATE_A] / baseMVA

        Pij_l = -Slmax
        Qij_l = -Slmax
        Iij_l = zeros(nl)
        Vm_l = bus[:, VMIN] ** 2
        Pg_l = gen[:, PMIN] / baseMVA
        Qg_l = gen[:, QMIN] / baseMVA

        Pij_u = Slmax
        Qij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Qg_u = 2 * gen[:, QMAX] / baseMVA

        nx = int(3 * nl + nb + 2 * ng + 2 * nmg + 2 * nb_traffic)
        self.nx = nx  # Number of decision variable within each time slot

        lx = concatenate([tile(concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l, Pmg_l, Qmg_l, Pev_l, Rev_l]), T)])
        ux = concatenate([tile(concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u, Pmg_u, Qmg_u, Pev_u, Rev_u]), T)])

        vtypes = ["c"] * nx * T
        NX = nx * T  # Number of total decision variables

        # Add system level constraints
        # 1) Active power balance
        Aeq_p = zeros((nb * T, NX))
        beq_p = zeros(nb * T)
        for i in range(T):
            Aeq_p[i * nb:(i + 1) * nb, i * nx: (i + 1) * nx] = hstack([Ct - Cf, zeros((nb, nl)),
                                                                       -diag(Ct * Branch_R) * Ct,
                                                                       zeros((nb, nb)), Cg,
                                                                       zeros((nb, ng)), -Cmg,
                                                                       zeros((nb, nmg)), Cev,
                                                                       zeros((nb, nb_traffic))]).toarray()

            beq_p[i * nb:(i + 1) * nb] = profile[i] * bus[:, PD] / baseMVA

        # 2) Reactive power balance
        Aeq_q = zeros((nb * T, NX))
        beq_q = zeros(nb * T)
        for i in range(T):
            Aeq_q[i * nb:(i + 1) * nb, i * nx: (i + 1) * nx] = hstack([zeros((nb, nl)), Ct - Cf,
                                                                       -diag(Ct * Branch_X) * Ct,
                                                                       zeros((nb, nb)),
                                                                       zeros((nb, ng)), Cg,
                                                                       zeros((nb, nmg)),
                                                                       -Cmg, zeros((nb, 2 * nb_traffic))]).toarray()
            beq_q[i * nb:(i + 1) * nb] = profile[i] * bus[:, QD] / baseMVA
        # 3) KVL equation
        Aeq_kvl = zeros((nl * T, NX))
        beq_kvl = zeros(nl * T)

        for i in range(T):
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx: i * nx + nl] = -2 * diag(Branch_R)
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx + nl: i * nx + 2 * nl] = -2 * diag(Branch_X)
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx + 2 * nl: i * nx + 3 * nl] = diag(Branch_R ** 2) + diag(Branch_X ** 2)
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx + 3 * nl:i * nx + 3 * nl + nb] = (Cf.T - Ct.T).toarray()

        Aeq = vstack([Aeq_p, Aeq_q, Aeq_kvl]).toarray()
        beq = concatenate([beq_p, beq_q, beq_kvl])

        # 4) Pij**2+Qij**2<=Vi*Iij
        Qc = dict()
        for t in range(T):
            for i in range(nl):
                Qc[t * nl + i] = [[int(t * nx + i), int(t * nx + i + nl),
                                   int(t * nx + i + 2 * nl), int(t * nx + f[i] + 3 * nl)],
                                  [int(t * nx + i), int(t * nx + i + nl),
                                   int(t * nx + f[i] + 3 * nl), int(t * nx + i + 2 * nl)],
                                  [1, 1, -1 / 2, -1 / 2]]

        c = zeros(NX)
        q = zeros(NX)
        c0 = 0
        for t in range(T):
            for i in range(ng):
                c[t * nx + i + 3 * nl + nb] = gencost[i, 5] * baseMVA
                q[t * nx + i + 3 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
                c0 += gencost[i, 6]

        # The boundary information
        Ax2y = zeros((2 * nmg * T, NX))  # connection matrix with the microgrids
        for i in range(T):
            for j in range(nmg):
                Ax2y[i * nmg + j, i * nx + 3 * nl + nb + 2 * ng + j] = 1000 * baseMVA  # Active power
                Ax2y[nmg * T + i * nmg + j,
                     i * nx + 3 * nl + nb + 2 * ng + nmg + j] = 1000 * baseMVA  # Reactive power

        Ax2z = zeros((2 * nb_traffic * T, NX))  # connection matrix with the tess
        for i in range(T):
            for j in range(nb_traffic):
                Ax2z[i * nb_traffic + j, i * nx + 3 * nl + nb + 2 * ng + 2 * nmg + j] = 1000 * baseMVA  # Active power
                Ax2z[nb_traffic * T + i * nb_traffic + j,
                     i * nx + 3 * nl + nb + 2 * ng + 2 * nmg + nb_traffic + j] = 1000 * baseMVA  # Spinning reserve

        # sol = miqcp(c, q, Aeq=Aeq, beq=beq, A=None, b=None, Qc=Qc, xmin=lx, xmax=ux)

        model_distribution_grid = {"c": c,
                                   "q": q,
                                   "lb": lx,
                                   "ub": ux,
                                   "vtypes": vtypes,
                                   "A": None,
                                   "b": None,
                                   "Aeq": Aeq,
                                   "beq": beq,
                                   "Qc": Qc,
                                   "c0": c0,
                                   "Ax2y": Ax2y,
                                   "Ax2z": Ax2z}

        return model_distribution_grid

    def solution_check_distribution_network(self, xx):
        """
        solution check for distribution networks solution
        :param xx:
        :return:
        """
        nl = self.nl
        nb = self.nb
        ng = self.ng
        T = self.T
        nx = self.nx
        nmg = self.nmg
        nb_traffic = self.nb_traffic
        f = self.f

        Pij = zeros((nl, T))
        Qij = zeros((nl, T))
        Iij = zeros((nl, T))
        Vi = zeros((nb, T))
        Pg = zeros((ng, T))
        Qg = zeros((ng, T))
        Pmg = zeros((nmg, T))
        Qmg = zeros((nmg, T))
        Pev = zeros((nb_traffic, T))
        Rev = zeros((nb_traffic, T))
        for i in range(T):
            Pij[:, i] = xx[i * nx:i * nx + nl]
            Qij[:, i] = xx[i * nx + nl: i * nx + 2 * nl]
            Iij[:, i] = xx[i * nx + 2 * nl:i * nx + 3 * nl]
            Vi[:, i] = xx[i * nx + 3 * nl: i * nx + 3 * nl + nb]
            Pg[:, i] = xx[i * nx + 3 * nl + nb: i * nx + 3 * nl + nb + ng]
            Qg[:, i] = xx[i * nx + 3 * nl + nb + ng: i * nx + 3 * nl + nb + 2 * ng]
            Pmg[:, i] = xx[i * nx + 3 * nl + nb + 2 * ng:i * nx + 3 * nl + nb + 2 * ng + nmg]
            Qmg[:, i] = xx[i * nx + 3 * nl + nb + 2 * ng + nmg:
                           i * nx + 3 * nl + nb + 2 * ng + 2 * nmg]
            Pev[:, i] = xx[i * nx + 3 * nl + nb + 2 * ng + 2 * nmg:
                           i * nx + 3 * nl + nb + 2 * ng + 2 * nmg + nb_traffic]
            Rev[:, i] = xx[i * nx + 3 * nl + nb + 2 * ng + 2 * nmg + nb_traffic:
                           i * nx + 3 * nl + nb + 2 * ng + 2 * nmg + 2 * nb_traffic]

        primal_residual = zeros((nl, T))
        for t in range(T):
            for i in range(nl):
                primal_residual[i, t] = Pij[i, t] * Pij[i, t] + Qij[i, t] * Qij[i, t] - Iij[i, t] * Vi[int(f[i]), t]

        sol = {"Pij": Pij,
               "Qij": Qij,
               "Iij": Iij,
               "Vi": Vi,
               "Pg": Pg,
               "Qg": Qg,
               "Pmg": Pmg,
               "Qmg": Qmg,
               "Pev": Pev,
               "Rev": Rev,
               "residual": primal_residual}

        return sol

    def problem_formulation_microgrid(self, micro_grid):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """

        try:
            T = self.T
        except:
            T = 24

        ## 1) boundary information and objective function
        nx = NX_MG * T
        lx = zeros(nx)
        ux = zeros(nx)
        c = zeros(nx)
        q = zeros(nx)
        vtypes = ["c"] * nx
        for i in range(T):
            ## 1.1) lower boundary
            lx[i * NX_MG + PG] = 0
            lx[i * NX_MG + QG] = micro_grid["DG"]["QMIN"]
            lx[i * NX_MG + RG] = 0
            lx[i * NX_MG + PUG] = 0
            lx[i * NX_MG + QUG] = micro_grid["UG"]["QMIN"]
            lx[i * NX_MG + RUG] = 0
            lx[i * NX_MG + PBIC_DC2AC] = 0
            lx[i * NX_MG + PBIC_AC2DC] = 0
            lx[i * NX_MG + QBIC] = -micro_grid["BIC"]["SMAX"]
            lx[i * NX_MG + PESS_CH] = 0
            lx[i * NX_MG + PESS_DC] = 0
            lx[i * NX_MG + RESS] = 0
            lx[i * NX_MG + EESS] = micro_grid["ESS"]["EMIN"]

            ## 1.2) upper boundary
            ux[i * NX_MG + PG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + QG] = micro_grid["DG"]["QMAX"]
            ux[i * NX_MG + RG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + PUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + QUG] = micro_grid["UG"]["QMAX"]
            ux[i * NX_MG + RUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + QBIC] = micro_grid["BIC"]["SMAX"]
            ux[i * NX_MG + PESS_CH] = micro_grid["ESS"]["PCH_MAX"]
            ux[i * NX_MG + PESS_DC] = micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + RESS] = micro_grid["ESS"]["PCH_MAX"] + micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + EESS] = micro_grid["ESS"]["EMAX"]

            ## 1.3) Objective functions
            c[i * NX_MG + PG] = micro_grid["DG"]["COST_A"]
            # c[i * NX_MG + PUG] = micro_grid["UG"]["COST"][i]

            ## 1.4) Upper and lower boundary information
            if i == T:
                lx[i * NX_MG + EESS] = micro_grid["ESS"]["E0"]
                ux[i * NX_MG + EESS] = micro_grid["ESS"]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = zeros((T, nx))
        beq = zeros(T)
        for i in range(T):
            Aeq[i, i * NX_MG + PG] = 1
            Aeq[i, i * NX_MG + PUG] = 1
            Aeq[i, i * NX_MG + PBIC_AC2DC] = -1
            Aeq[i, i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["EFF_DC2AC"]
            beq[i] = micro_grid["PD"]["AC"][i]
        # b) DC bus equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["EFF_AC2DC"]
            Aeq_temp[i, i * NX_MG + PBIC_DC2AC] = -1
            Aeq_temp[i, i * NX_MG + PESS_CH] = -1
            Aeq_temp[i, i * NX_MG + PESS_DC] = 1
            beq_temp[i] = micro_grid["PD"]["DC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # c) AC reactive power balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + QUG] = 1
            Aeq_temp[i, i * NX_MG + QBIC] = 1
            Aeq_temp[i, i * NX_MG + QG] = 1
            beq_temp[i] = micro_grid["QD"]["AC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + EESS] = 1
            Aeq_temp[i, i * NX_MG + PESS_CH] = -micro_grid["ESS"]["EFF_CH"]
            Aeq_temp[i, i * NX_MG + PESS_DC] = 1 / micro_grid["ESS"]["EFF_DC"]
            if i == 0:
                beq_temp[i] = micro_grid["ESS"]["E0"]
            else:
                Aeq_temp[i, (i - 1) * NX_MG + EESS] = -1
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequal constraints
        # 3.1) Pg+Rg<=Ig*Pgmax
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(T):
            A[i, i * NX_MG + PG] = 1
            A[i, i * NX_MG + RG] = 1
            b[i] = micro_grid["DG"]["PMAX"]
        # 3.2) Pg-Rg>=Ig*Pgmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + RG] = 1
            A_temp[i, i * NX_MG + PG] = -1
            b_temp[i] = -micro_grid["DG"]["PMIN"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.3) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PESS_DC] = 1
            A_temp[i, i * NX_MG + PESS_CH] = -1
            A_temp[i, i * NX_MG + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PDC_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.4) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PESS_CH] = 1
            A_temp[i, i * NX_MG + PESS_DC] = -1
            A_temp[i, i * NX_MG + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PCH_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.5) Pug+Rug<=Iug*Pugmax
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PUG] = 1
            A_temp[i, i * NX_MG + RUG] = 1
            b_temp[i] = micro_grid["UG"]["PMAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.6) Pug-Rug>=Iug*Pugmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + RUG] = 1
            A_temp[i, i * NX_MG + PUG] = -1
            b_temp[i] = -micro_grid["DG"]["PMIN"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        # sol = milp(c, q=q, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lx, xmax=ux)

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lx,
                            "ub": ux,
                            "vtypes": vtypes,
                            "A": A,
                            "b": b,
                            "Aeq": Aeq,
                            "beq": beq,
                            "NX": NX_MG,
                            "PG": PG,
                            "QG": QG}

        return model_micro_grid

    def solution_check_microgrids(self, xx, nVariables_index):
        T = self.T
        nmg = self.nmg

        Pess_dc = zeros((nmg, T))
        Pess_ch = zeros((nmg, T))
        Ress = zeros((nmg, T))
        Eess = zeros((nmg, T))
        # b) Diesel generator group
        Pg = zeros((nmg, T))
        Qg = zeros((nmg, T))
        Rg = zeros((nmg, T))
        # c) Utility grid group
        Pug = zeros((nmg, T))
        Qug = zeros((nmg, T))
        Rug = zeros((nmg, T))
        # d) Bi-directional converter group
        Pbic_a2d = zeros((nmg, T))
        Pbic_d2a = zeros((nmg, T))
        Qbic = zeros((nmg, T))
        for i in range(T):
            for j in range(nmg):
                Pess_dc[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PESS_DC]
                Pess_ch[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PESS_CH]
                Ress[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RESS]
                Eess[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + EESS]

                Pg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PG]
                Qg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QG]
                Rg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RG]

                Pug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PUG]
                Qug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QUG]
                Rug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RUG]

                Pbic_a2d[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PBIC_AC2DC]
                Pbic_d2a[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PBIC_DC2AC]
                Qbic[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QBIC]
        # e) voilation of bi-directional power flows
        vol_bic = zeros((nmg, T))
        vol_ess = zeros((nmg, T))
        for i in range(T):
            for j in range(nmg):
                vol_ess[j, i] = Pess_dc[j, i] * Pess_ch[j, i]
                vol_bic[j, i] = Pbic_a2d[j, i] * Pbic_d2a[j, i]

        sol_microgrids = {"PESS_DC": Pess_dc,
                          "PESS_CH": Pess_ch,
                          "RESS": Ress,
                          "EESS": Eess,
                          "PG": Pg,
                          "QG": Qg,
                          "RG": Rg,
                          "PUG": Pug,
                          "QUG": Qug,
                          "RUG": Rug,
                          "PBIC_AC2DC": Pbic_a2d,
                          "PBIC_DC2AC": Pbic_d2a,
                          "QBIC": Qbic,
                          "VOL_BIC": vol_bic,
                          "VOL_ESS": vol_ess, }

        return sol_microgrids

    def problem_formulation_tess(self, tess, traffic_networks):
        """
        Problem formulation for transportation energy storage scheduling, including vehicle routine problem and etc.
        :param tess: specific tess information
        :param traffic_network: transportation network information
        :return:
        """
        nb_traffic = self.nb_traffic
        T = self.T
        nb = self.nb

        nl_traffic = traffic_networks["branch"].shape[0]

        # Formulate the connection matrix between the transportaion networks and power networks
        connection_matrix = zeros(((2 * nl_traffic + nb_traffic) * T, 4))
        weight = zeros(((2 * nl_traffic + nb_traffic) * T, 1))
        for i in range(T):
            for j in range(nl_traffic):
                # Add from matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, F_BUS] = traffic_networks["branch"][j, F_BUS] + \
                                                                                  i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, T_BUS] = traffic_networks["branch"][j, T_BUS] + \
                                                                                  traffic_networks["branch"][j, TIME] * \
                                                                                  nb_traffic + i * nb_traffic
                weight[i * (2 * nl_traffic + nb_traffic) + j, 0] = 1
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, TIME] = traffic_networks["branch"][j, TIME]

            for j in range(nl_traffic):
                # Add to matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, F_BUS] = \
                    traffic_networks["branch"][j, T_BUS] + i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, T_BUS] = \
                    traffic_networks["branch"][j, F_BUS] + traffic_networks["branch"][j, TIME] * nb_traffic + \
                    i * nb_traffic

                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, TIME] = \
                    traffic_networks["branch"][j, TIME]

            for j in range(nb_traffic):
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, F_BUS] = \
                    j + i * nb_traffic  # This time slot
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, T_BUS] = \
                    j + (i + 1) * nb_traffic  # The next time slot

                if traffic_networks["bus"][j, LOCATION] >= 0:
                    connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, 3] = \
                        traffic_networks["bus"][j, LOCATION] + i * nb  # Location information

        # Delete the out of range lines
        index = find(connection_matrix[:, T_BUS] < T * nb_traffic)
        connection_matrix = connection_matrix[index, :]

        # add two virtual nodes to represent the initial and end status of vehicles
        # special attention should be paid here, as the original index has been modified!
        connection_matrix[:, F_BUS] += 1
        connection_matrix[:, T_BUS] += 1
        # From matrix
        temp = zeros((nb_traffic, 4))
        for i in range(nb_traffic): temp[i, 1] = i + 1
        connection_matrix = concatenate([temp, connection_matrix])

        # To matrix
        for i in range(nb_traffic):
            temp = zeros((1, 4))
            temp[0, 0] = nb_traffic * (T - 1) + i + 1
            temp[0, 1] = nb_traffic * T + 1
            if traffic_networks["bus"][i, LOCATION] >= 0:
                temp[0, 3] = traffic_networks["bus"][i, LOCATION] + (T - 1) * nb
            connection_matrix = concatenate([connection_matrix, temp])

        # Status transition matrix
        nl_traffic = connection_matrix.shape[0]
        nb_traffic_electric = sum((traffic_networks["bus"][:, 2]) >= 0)
        # 0 represents that, the bus is not within the power networks

        status_matrix = zeros((T, nl_traffic))
        for i in range(T):
            for j in range(nl_traffic):
                if connection_matrix[j, F_BUS] >= i * nb_traffic + 1 and \
                        connection_matrix[j, F_BUS] < (i + 1) * nb_traffic + 1:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb_traffic + 1 and \
                        connection_matrix[j, T_BUS] > (i + 1) * nb_traffic + 1:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = zeros((T * nb_traffic + 2, nl_traffic))
        connection_matrix_t = zeros((T * nb_traffic + 2, nl_traffic))

        for i in range(T * nb_traffic + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        n_stops = find(connection_matrix[:, 3]).__len__()

        assert n_stops == nb_traffic_electric * T, "The number of bus stop is not right!"

        NX_traffic = nl_traffic + 4 * n_stops  # Status transition, charging status, charging rate, discharging rate, spinning reserve
        NX_status = nl_traffic
        lx = zeros(NX_traffic)
        ux = ones(NX_traffic)

        self.NX_traffic = NX_traffic
        self.nl_traffic = nl_traffic
        self.n_stops = n_stops

        ux[NX_status + 0 * n_stops:NX_status + 1 * n_stops] = 1
        ux[NX_status + 1 * n_stops:NX_status + 2 * n_stops] = tess["PDMAX"]
        ux[NX_status + 2 * n_stops:NX_status + 3 * n_stops] = tess["PCMAX"]
        ux[NX_status + 3 * n_stops:NX_status + 4 * n_stops] = tess["PCMAX"] + tess["PDMAX"]
        # The initial location and stop location
        lx[find(connection_matrix[:, F_BUS] == 0)] = tess["initial"]
        ux[find(connection_matrix[:, F_BUS] == 0)] = tess["initial"]
        lx[find(connection_matrix[:, T_BUS] == T * nb_traffic + 1)] = tess["end"]
        ux[find(connection_matrix[:, T_BUS] == T * nb_traffic + 1)] = tess["end"]

        vtypes = ["b"] * NX_status + ["b"] * n_stops + ["c"] * 3 * n_stops

        Aeq = connection_matrix_f - connection_matrix_t
        beq = zeros(T * nb_traffic + 2)
        beq[0] = 1
        beq[-1] = -1
        # statue constraints
        Aeq_temp = status_matrix
        beq_temp = ones(T)
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        neq_traffic = Aeq.shape[0]
        # Fulfill the missing zeros
        Aeq = concatenate([Aeq, zeros((neq_traffic, 4 * n_stops))], axis=1)

        ## Inequality constraints
        index_stops = find(connection_matrix[:, 3])
        index_operation = arange(n_stops)
        power_limit = sparse((ones(n_stops), (index_operation, index_stops)), (n_stops, NX_status))
        # This mapping matrix plays an important role in the connection between the power network and traffic network
        ## 1) Stopping status
        A = zeros((3 * n_stops, NX_traffic))  # Charging, discharging status,RBS
        # Discharging
        A[0:n_stops, 0: NX_status] = -power_limit.toarray() * tess["PDMAX"]
        A[0:n_stops, NX_status + n_stops: NX_status + 2 * n_stops] = eye(n_stops)
        # Charging
        A[n_stops:n_stops * 2, 0: NX_status] = -power_limit.toarray() * tess["PCMAX"]

        A[n_stops:n_stops * 2, NX_status + 2 * n_stops:NX_status + 3 * n_stops] = eye(n_stops)
        # spinning reserve
        A[n_stops * 2: n_stops * 3, 0: NX_status] = -power_limit.toarray() * (tess["PCMAX"] + tess["PDMAX"])
        A[n_stops * 2:n_stops * 3, NX_status + 3 * n_stops:NX_status + 4 * n_stops] = eye(n_stops)

        b = zeros(3 * n_stops)

        ## 2) Operating status
        Arange = zeros((2 * n_stops, NX_traffic))
        brange = zeros(2 * n_stops)
        # 1) Pdc<(1-Ic)*Pdc_max
        Arange[0: n_stops, NX_status:NX_status + n_stops] = eye(n_stops) * tess["PDMAX"]
        Arange[0: n_stops, NX_status + n_stops: NX_status + n_stops * 2] = eye(n_stops)
        brange[0: n_stops] = ones(n_stops) * tess["PDMAX"]
        # 2) Pc<Ic*Pch_max
        Arange[n_stops:n_stops * 2, NX_status: NX_status + n_stops] = -eye(n_stops) * tess["PCMAX"]
        Arange[n_stops:n_stops * 2, NX_status + n_stops * 2: NX_status + n_stops * 3] = eye(n_stops)
        A = concatenate([A, Arange])
        b = concatenate([b, brange])

        ## 2) Power limitation
        Areserve = zeros((2 * n_stops, NX_traffic))
        breserve = zeros(2 * n_stops)
        # 1) Pdc-Pc+Rbs<=Pdc_max
        Areserve[0: n_stops, NX_status + n_stops: NX_status + n_stops * 2] = eye(n_stops)
        Areserve[0: n_stops, NX_status + n_stops * 2:NX_status + n_stops * 3] = -eye(n_stops)
        Areserve[0: n_stops, NX_status + n_stops * 3:NX_status + n_stops * 4] = eye(n_stops)
        breserve[0: n_stops] = ones(n_stops) * tess["PDMAX"]
        # 2) Pc-Pdc<=Pc_max
        Areserve[n_stops:n_stops * 2, NX_status + n_stops: NX_status + n_stops * 2] = -eye(n_stops)
        Areserve[n_stops:n_stops * 2, NX_status + n_stops * 2:NX_status + n_stops * 3] = eye(n_stops)
        breserve[n_stops:n_stops * 2] = ones(n_stops) * tess["PCMAX"]

        A = concatenate([A, Areserve])
        b = concatenate([b, breserve])

        # Add constraints on the energy status
        Aenergy = zeros((2 * T, NX_traffic))
        benergy = zeros(2 * T)
        for j in range(T):
            # minimal energy
            Aenergy[j, NX_status + n_stops: NX_status + n_stops + (j + 1) * nb_traffic_electric] = 1 / tess["EFF_DC"]
            Aenergy[j, NX_status + 2 * n_stops: NX_status + 2 * n_stops + (j + 1) * nb_traffic_electric] = \
                -tess["EFF_CH"]
            Aenergy[j, NX_status + 3 * n_stops + (j + 1) * nb_traffic_electric - 1] = 0.5
            if j != (T - 1):
                benergy[j] = tess["E0"] - tess["EMIN"]
            else:
                benergy[j] = 0
            # maximal energy
            Aenergy[T + j, NX_status + n_stops: NX_status + n_stops + (j + 1) * nb_traffic_electric] = \
                -1 / tess["EFF_DC"]
            Aenergy[T + j, NX_status + 2 * n_stops: NX_status + 2 * n_stops +
                                                    (j + 1) * nb_traffic_electric] = tess["EFF_CH"]
            if j != (T - 1):
                benergy[T + j] = tess["EMAX"] - tess["E0"]
            else:
                benergy[T + j] = 0

        A = concatenate([A, Aenergy])
        b = concatenate([b, benergy])

        # sol = milp(zeros(NX_traffic), q=zeros(NX_traffic), Aeq=Aeq, beq=beq, A=A, b=b, xmin=lx, xmax=ux)

        model_tess = {"c": zeros(NX_traffic),
                      "q": zeros(NX_traffic),
                      "lb": lx,
                      "ub": ux,
                      "vtypes": vtypes,
                      "A": A,
                      "b": b,
                      "Aeq": Aeq,
                      "beq": beq,
                      "NX": NX_traffic, }

        return model_tess

    def solution_check_tess(self, sol):
        """
        :param sol: solutions for tess
        :return: decoupled solutions for tess
        """

        NX_traffic = self.NX_traffic
        nl_traffic = self.nl_traffic
        n_stops = self.n_stops
        nev = self.nev
        T = self.T

        tsn_ev = zeros((nl_traffic, nev))
        ich_ev = zeros((n_stops, nev))
        pdc_ev = zeros((n_stops, nev))
        pch_ev = zeros((n_stops, nev))
        rs_ev = zeros((n_stops, nev))

        for i in range(nev):
            for j in range(nl_traffic):
                tsn_ev[j, i] = sol[i * NX_traffic + j]
            for j in range(n_stops):
                ich_ev[j, i] = sol[i * NX_traffic + nl_traffic + 0 * n_stops + j]
            for j in range(n_stops):
                pdc_ev[j, i] = sol[i * NX_traffic + nl_traffic + 1 * n_stops + j]
            for j in range(n_stops):
                pch_ev[j, i] = sol[i * NX_traffic + nl_traffic + 2 * n_stops + j]
            for j in range(n_stops):
                rs_ev[j, i] = sol[i * NX_traffic + nl_traffic + 3 * n_stops + j]

        sol_tess = {"Tsn_ev": tsn_ev,
                    "Ich": ich_ev,
                    "Pdc": pdc_ev,
                    "Pch": pch_ev,
                    "Rs": rs_ev, }

        return sol_tess


if __name__ == "__main__":
    # Distribution network information
    mpc = case33.case33()  # Default test case
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41]) * 2

    # Microgrid information
    Profile = array([
        [0.64, 0.63, 0.65, 0.64, 0.66, 0.69, 0.75, 0.91, 0.95, 0.97, 1.00, 0.97, 0.97, 0.95, 0.98, 0.99, 0.95, 0.95,
         0.94, 0.95, 0.97, 0.93, 0.85, 0.69],
        [0.78, 0.75, 0.74, 0.74, 0.75, 0.81, 0.91, 0.98, 0.99, 0.99, 1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.95,
         0.95, 0.95, 0.96, 0.95, 0.88, 0.82],
        [0.57, 0.55, 0.55, 0.56, 0.62, 0.70, 0.78, 0.83, 0.84, 0.89, 0.87, 0.82, 0.80, 0.80, 0.84, 0.89, 0.94, 0.98,
         1.00, 0.97, 0.87, 0.79, 0.72, 0.62]
    ])
    micro_grid_1 = deepcopy(micro_grid)
    micro_grid_1["BUS"] = 2
    micro_grid_1["PD"]["AC_MAX"] = 50
    micro_grid_1["PD"]["DC_MAX"] = 50
    micro_grid_1["UG"]["PMIN"] = -500
    micro_grid_1["UG"]["PMAX"] = 500
    micro_grid_1["UG"]["QMIN"] = -500
    micro_grid_1["UG"]["QMAX"] = 500
    micro_grid_1["DG"]["PMAX"] = 100
    micro_grid_1["DG"]["QMAX"] = 100
    micro_grid_1["DG"]["QMIN"] = -100
    micro_grid_1["DG"]["COST_A"] = 0.015
    micro_grid_1["ESS"]["PDC_MAX"] = 50
    micro_grid_1["ESS"]["PCH_MAX"] = 50
    micro_grid_1["ESS"]["E0"] = 50
    micro_grid_1["ESS"]["EMIN"] = 10
    micro_grid_1["ESS"]["EMAX"] = 100
    micro_grid_1["BIC"]["PMAX"] = 100
    micro_grid_1["BIC"]["QMAX"] = 100
    micro_grid_1["BIC"]["SMAX"] = 100
    micro_grid_1["PD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_1["QD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"] * 0.2
    micro_grid_1["PD"]["DC"] = Profile[0] * micro_grid_1["PD"]["DC_MAX"]
    # micro_grid_1["MG"]["PMIN"] = 0
    # micro_grid_1["MG"]["PMAX"] = 0

    micro_grid_2 = deepcopy(micro_grid)
    micro_grid_2["BUS"] = 4
    micro_grid_2["PD"]["AC_MAX"] = 50
    micro_grid_2["PD"]["DC_MAX"] = 50
    micro_grid_2["UG"]["PMIN"] = -500
    micro_grid_2["UG"]["PMAX"] = 500
    micro_grid_1["UG"]["QMIN"] = -500
    micro_grid_1["UG"]["QMAX"] = 500
    micro_grid_2["DG"]["PMAX"] = 50
    micro_grid_1["DG"]["QMAX"] = 50
    micro_grid_1["DG"]["QMIN"] = -50
    micro_grid_2["DG"]["COST_A"] = 0.01
    micro_grid_2["ESS"]["PDC_MAX"] = 50
    micro_grid_2["ESS"]["PCH_MAX"] = 50
    micro_grid_2["ESS"]["E0"] = 15
    micro_grid_2["ESS"]["EMIN"] = 10
    micro_grid_2["ESS"]["EMAX"] = 50
    micro_grid_2["BIC"]["PMAX"] = 100
    micro_grid_2["BIC"]["QMAX"] = 100
    micro_grid_2["BIC"]["SMAX"] = 100
    micro_grid_2["PD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"]
    micro_grid_2["QD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"] * 0.2
    micro_grid_2["PD"]["DC"] = Profile[1] * micro_grid_2["PD"]["DC_MAX"]
    # micro_grid_2["MG"]["PMIN"] = 0
    # micro_grid_2["MG"]["PMAX"] = 0

    micro_grid_3 = deepcopy(micro_grid)
    micro_grid_3["BUS"] = 10
    micro_grid_3["PD"]["AC_MAX"] = 50
    micro_grid_3["PD"]["DC_MAX"] = 50
    micro_grid_3["UG"]["PMIN"] = -500
    micro_grid_3["UG"]["PMAX"] = 500
    micro_grid_3["UG"]["QMIN"] = -500
    micro_grid_3["UG"]["QMAX"] = 500
    micro_grid_3["DG"]["PMAX"] = 50
    micro_grid_3["DG"]["QMAX"] = 50
    micro_grid_3["DG"]["QMIN"] = -50
    micro_grid_3["DG"]["COST_A"] = 0.01
    micro_grid_3["ESS"]["PDC_MAX"] = 50
    micro_grid_3["ESS"]["PCH_MAX"] = 50
    micro_grid_3["ESS"]["E0"] = 20
    micro_grid_3["ESS"]["EMIN"] = 10
    micro_grid_3["ESS"]["EMAX"] = 50
    micro_grid_3["BIC"]["PMAX"] = 50
    micro_grid_3["BIC"]["QMAX"] = 100
    micro_grid_3["BIC"]["SMAX"] = 100
    micro_grid_3["PD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"]
    micro_grid_3["QD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"] * 0.2
    micro_grid_3["PD"]["DC"] = Profile[2] * micro_grid_3["PD"]["DC_MAX"]
    case_micro_grids = [micro_grid_1, micro_grid_2, micro_grid_3]

    ## Transportaion network information
    ev = []
    traffic_networks = case3.transportation_network()  # Default transportation networks
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    """
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 1, 0]),
               "PCMAX": 1000,
               "PDMAX": 1000,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    """

    dynamic_optimal_power_flow = DynamicOptimalPowerFlowTess()

    (sol_dso, sol_mgs, sol_tess) = dynamic_optimal_power_flow.main(case=mpc, profile=load_profile.tolist(),
                                                                   microgrids=case_micro_grids, tess=ev,
                                                                   traffic_networks=traffic_networks)

    print(max(sol_dso["residual"][0]))
