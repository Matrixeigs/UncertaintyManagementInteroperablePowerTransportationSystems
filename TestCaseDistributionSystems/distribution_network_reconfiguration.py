"""
Distribution network reconfiguration considering stochastic disturbance of renewable sources and loads

The modelling language is based on Gurobi Python interface
"""

from gurobipy import *

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from numpy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack
from numpy import flatnonzero as find

from distribution_system_optimization.test_cases import case33


class DynamicNetworkReconfiguration(Model):
    """
    Network reconfiguration model for AC distribution systems
    including the following three steps:
    1)
    """

    def __init__(self):

        super(DynamicNetworkReconfiguration, self).__init__()

        self.Params.OutputFlag = 0
        self.Params.LogToConsole = 0
        self.Params.DisplayInterval = 1
        self.Params.LogFile = ""
        self._nl = 0
        self._nb = 0
        self._ng = 0

    def problem_formulation(self, case):
        """
        problem formulation for distribution network reformulation
        :param case:
        :return:
        """
        case["branch"][:, BR_STATUS] = ones(case["branch"].shape[0])
        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
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
        Alpha_l = zeros(nl)
        Beta_f_l = zeros(nl)
        Beta_t_l = zeros(nl)

        Pij_u = Slmax
        Qij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Qg_u = 2 * gen[:, QMAX] / baseMVA
        Alpha_u = ones(nl)
        Beta_f_u = ones(nl)
        Beta_t_u = ones(nl)
        bigM = max(Vm_u)
        # For the spanning tree constraints
        Root_node = find(bus[:, BUS_TYPE] == REF)
        Root_line = find(branch[:, F_BUS] == Root_node)

        Span_f = zeros((nb, nl))
        Span_t = zeros((nb, nl))
        for i in range(nb):
            Span_f[i, find(branch[:, F_BUS] == i)] = 1
            Span_t[i, find(branch[:, T_BUS] == i)] = 1

        Alpha_l[Root_line] = 1
        Alpha_u[Root_line] = 1
        Beta_f_l[Root_line] = 0
        Beta_f_l[Root_line] = 0

        lx = concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l, Alpha_l, Beta_f_l, Beta_t_l])
        ux = concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u, Alpha_u, Beta_f_u, Beta_t_u])
        vtypes = ["c"] * (3 * nl + nb + 2 * ng + nl) + ["b"] * 2 * nl

        # Define the decision variables
        x = {}
        nx = lx.shape[0]

        for i in range(nx):
            if vtypes[i] == "c":
                x[i] = self.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
            elif vtypes[i] == "b":
                x[i] = self.addVar(lb=lx[i], ub=ux[i], vtype=GRB.BINARY)

        # Alpha = Beta_f + Beta_t
        Aeq_f = zeros((nl, nx))
        beq_f = zeros(nl)
        Aeq_f[:, 3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl] = -eye(nl)
        Aeq_f[:, 3 * nl + nb + 2 * ng + nl:3 * nl + nb + 2 * ng + 2 * nl] = eye(nl)
        Aeq_f[:, 3 * nl + nb + 2 * ng + 2 * nl:3 * nl + nb + 2 * ng + 3 * nl] = eye(nl)

        # sum(alpha)=nb-1
        Aeq_alpha = zeros((1, nx))
        beq_alpha = zeros(1)
        Aeq_alpha[0, 3 * nl + nb + 2 * ng: 3 * nl + nb + 2 * ng + nl] = ones(nl)
        beq_alpha[0] = nb - 1

        # Span_f*Beta_f+Span_t*Beta_t = Spanning_tree
        Aeq_span = zeros((nb, nx))
        beq_span = ones(nb)
        beq_span[Root_node] = 0
        Aeq_span[:, 3 * nl + nb + 2 * ng + nl:3 * nl + nb + 2 * ng + 2 * nl] = Span_f
        Aeq_span[:, 3 * nl + nb + 2 * ng + 2 * nl:] = Span_t

        # Add system level constraints
        Aeq_p = hstack(
            [Ct - Cf, zeros((nb, nl)), -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, zeros((nb, ng + 3 * nl))])
        beq_p = bus[:, PD] / baseMVA
        # Add constraints for each sub system
        Aeq_q = hstack([zeros((nb, nl)), Ct - Cf, -diag(Ct * Branch_X) * Ct, zeros((nb, nb)), zeros((nb, ng)), Cg,
                        zeros((nb, 3 * nl))])
        beq_q = bus[:, QD] / baseMVA

        Aeq = vstack([Aeq_f, Aeq_alpha, Aeq_span, Aeq_p, Aeq_q])
        Aeq = Aeq.toarray()
        beq = concatenate([beq_f, beq_alpha, beq_span, beq_p, beq_q])

        # Inequality constraints
        A = zeros((nl, nx))
        b = zeros(nl)
        A[:, 2 * nl:3 * nl] = eye(nl)
        A[:, 3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl] = -diag(Iij_u)

        A_temp = zeros((nl, nx))
        b_temp = zeros(nl)
        A_temp[:, 0: nl] = eye(nl)
        A_temp[:, 3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl] = -diag(Pij_u)
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        #
        A_temp = zeros((nl, nx))
        b_temp = zeros(nl)
        A_temp[:, nl: 2 * nl] = eye(nl)
        A_temp[:, 3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl] = -diag(Qij_u)
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((nl, nx))
        b_temp = zeros(nl)
        A_temp[:, 0:nl] = -2 * diag(Branch_R)
        A_temp[:, nl:2 * nl] = -2 * diag(Branch_X)
        A_temp[:, 2 * nl:3 * nl] = diag(Branch_R ** 2) + diag(Branch_X ** 2)
        A_temp[:, 3 * nl:3 * nl + nb] = (Cf.T - Ct.T).toarray()
        A_temp[:, 3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl] = eye(nl) * bigM
        b_temp = ones(nl) * bigM
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((nl, nx))
        b_temp = zeros(nl)
        A_temp[:, 0:nl] = 2 * diag(Branch_R)
        A_temp[:, nl:2 * nl] = 2 * diag(Branch_X)
        A_temp[:, 2 * nl:3 * nl] = -diag(Branch_R ** 2) - diag(Branch_X ** 2)
        A_temp[:, 3 * nl:3 * nl + nb] = (-Cf.T + Ct.T).toarray()
        A_temp[:, 3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl] = eye(nl) * bigM
        b_temp = ones(nl) * bigM
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        neq = len(beq)
        for i in range(neq):
            expr = 0
            for j in range(nx):
                expr += x[j] * Aeq[i, j]
            self.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

        nineq = len(b)
        for i in range(nineq):
            expr = 0
            for j in range(nx):
                expr += x[j] * A[i, j]
            self.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b[i])

        for i in range(nl):
            self.addConstr(x[i] * x[i] + x[i + nl] * x[i + nl] <= x[i + 2 * nl] * x[f[i] + 3 * nl],
                           name='"rc{0}"'.format(i))

        obj = 0
        for i in range(ng):
            obj += gencost[i, 4] * x[i + 3 * nl + nb] * x[i + 3 * nl + nb] * baseMVA * baseMVA + gencost[i, 5] * x[
                i + 3 * nl + nb] * baseMVA + gencost[i, 6]

        self.setObjective(obj)

        self._nl = nl
        self._nb = nb
        self._ng = ng
        self._f = f

    def problem_solving(self):
        """
        problem
        :return:
        """
        self.optimize()
        nl = self._nl
        nb = self._nb
        ng = self._ng

        xx = []
        for v in self.getVars():
            xx.append(v.x)

        Pij = xx[0:nl]
        Qij = xx[nl + 0:2 * nl]
        Iij = xx[2 * nl:3 * nl]
        Vi = xx[3 * nl:3 * nl + nb]
        Pg = xx[3 * nl + nb:3 * nl + nb + ng]
        Qg = xx[3 * nl + nb + ng:3 * nl + nb + 2 * ng]
        Alpha = xx[3 * nl + nb + 2 * ng:3 * nl + nb + 2 * ng + nl]
        Beta_f = xx[3 * nl + nb + 2 * ng + nl:3 * nl + nb + 2 * ng + 2 * nl]
        Beta_t = xx[3 * nl + nb + 2 * ng + 2 * nl:3 * nl + nb + 2 * ng + 3 * nl]

        primal_residual = zeros(nl)

        for i in range(nl):
            primal_residual[i] = Pij[i] * Pij[i] + Qij[i] * Qij[i] - Iij[i] * Vi[int(self._f[i])]

        self._sol = {"Pij": Pij,
                     "Qij": Qij,
                     "Iij": Iij,
                     "Vi": Vi,
                     "Pg": Pg,
                     "Qg": Qg,
                     "Alpha": Alpha,
                     "Beta_f": Beta_f,
                     "Beta_t": Beta_t,
                     "residual": primal_residual,
                     "obj": self.ObjVal}

        assert max(primal_residual) < 10 ** -4

    def solution_check(self):
        """

        :return:
        """


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    network_reconfiguration = DynamicNetworkReconfiguration()

    # network_reconfiguration.parameter_setting()
    network_reconfiguration.problem_formulation(case=mpc)

    network_reconfiguration.problem_solving()

    print(max(network_reconfiguration._sol["residual"]))
