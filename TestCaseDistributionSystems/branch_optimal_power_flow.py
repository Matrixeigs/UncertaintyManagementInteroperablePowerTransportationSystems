"""
Branch optimal power flow for radial networks
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""
from TestCaseDistributionSystems.test_cases import case33
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int


class BranchOptimalPowerFlow():
    def __init__(self):
        self.name = "Optimal power flow"

    def main(self, case):
        """
        Modelling and simulation of
        :param case:
        :return:
        """
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

        Pij_u = Slmax
        Qij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Qg_u = 2 * gen[:, QMAX] / baseMVA
        lx = concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l])
        ux = concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u])
        model = Model("OPF")
        # Define the decision variables
        x = {}
        nx = 3 * nl + nb + 2 * ng
        for i in range(nx):
            x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)

        # Add system level constraints
        Aeq_p = hstack([Ct - Cf, zeros((nb, nl)), -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, zeros((nb, ng))])
        beq_p = bus[:, PD] / baseMVA
        # Add constraints for each sub system
        Aeq_q = hstack([zeros((nb, nl)), Ct - Cf, -diag(Ct * Branch_X) * Ct, zeros((nb, nb)), zeros((nb, ng)), Cg])
        beq_q = bus[:, QD] / baseMVA
        Aeq_KVL = hstack([-2 * diags(Branch_R), -2 * diags(Branch_X),
                          diags(Branch_R ** 2) + diags(Branch_X ** 2), Cf.T - Ct.T,
                          zeros((nl, 2 * ng))])
        beq_KVL = zeros(nl)

        Aeq = vstack([Aeq_p, Aeq_q, Aeq_KVL])
        Aeq = Aeq.todense()
        beq = concatenate([beq_p, beq_q, beq_KVL])
        neq = len(beq)

        for i in range(neq):
            expr = 0
            for j in range(nx):
                expr += x[j] * Aeq[i, j]
            model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

        for i in range(nl):
            model.addConstr(x[i] * x[i] + x[i + nl] * x[i + nl] <= x[i + 2 * nl] * x[f[i] + 3 * nl],
                            name='"rc{0}"'.format(i))

        obj = 0
        for i in range(ng):
            obj += gencost[i, 4] * x[i + 3 * nl + nb] * x[i + 3 * nl + nb] * baseMVA * baseMVA + gencost[i, 5] * x[
                i + 3 * nl + nb] * baseMVA + gencost[i, 6]

        model.setObjective(obj)
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        model.Params.DisplayInterval = 1
        model.Params.LogFile = ""

        model.optimize()

        xx = []
        for v in model.getVars():
            xx.append(v.x)

        obj = obj.getValue()

        Pij = xx[0:nl]
        Qij = xx[nl + 0:2 * nl]
        Iij = xx[2 * nl:3 * nl]
        Vi = xx[3 * nl:3 * nl + nb]
        Pg = xx[3 * nl + nb:3 * nl + nb + ng]
        Qg = xx[3 * nl + nb + ng:3 * nl + nb + 2 * ng]

        primal_residual = zeros((nl, 1))

        for i in range(nl):
            primal_residual[i] = Pij[i] * Pij[i] + Qij[i] * Qij[i] - Iij[i] * Vi[int(f[i])]

        return xx, obj, primal_residual


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    branch_optimal_power = BranchOptimalPowerFlow()

    (xx, obj, residual) = branch_optimal_power.main(case=mpc)

    # result = runopf.runopf(case33.case33())

    # gap = 100 * (result["f"] - obj) / obj

    # print(gap)
    print(max(residual))
