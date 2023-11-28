"""

Optimal power flow based on branch power flow modelling

Additional case33 is added to the test cases

"""

from distribution_system_optimization.test_cases import case33
from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags


def run(mpc):
    """
    Gurobi based optimal power flow modelling and solution
    :param mpc: The input case of optimal power flow
    :return: obtained solution
    """
    # Data format
    from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
    from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
    from pypower.ext2int import ext2int

    mpc = ext2int(mpc)
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
    Vm_l = turn_to_power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA
    Qg_l = gen[:, QMIN] / baseMVA

    Pij_u = Slmax
    Qij_u = Slmax
    Iij_u = Slmax
    Vm_u = turn_to_power(bus[:, VMAX], 2)
    Pg_u = 2 * gen[:, PMAX] / baseMVA
    Qg_u = 2 * gen[:, QMAX] / baseMVA

    model = Model("OPF")
    # Define the decision variables
    x = {}

    for i in range(nl):
        x[i] = model.addVar(lb=Pij_l[i], ub=Pij_u[i], vtype=GRB.CONTINUOUS, name='"Pij{0}"'.format(i))
        x[i + nl] = model.addVar(lb=Qij_l[i], ub=Qij_u[i], vtype=GRB.CONTINUOUS, name='"Qij{0}"'.format(i))
        x[i + 2 * nl] = model.addVar(lb=Iij_l[i], ub=Iij_u[i], vtype=GRB.CONTINUOUS, name='"Iij{0}"'.format(i))

    for i in range(nb):
        x[i + 3 * nl] = model.addVar(lb=Vm_l[i], ub=Vm_u[i], vtype=GRB.CONTINUOUS, name='"V{0}"'.format(i))

    for i in range(ng):
        x[i + 3 * nl + nb] = model.addVar(lb=Pg_l[i], ub=Pg_u[i], vtype=GRB.CONTINUOUS, name='"Pg{0}"'.format(i))
        x[i + 3 * nl + nb + ng] = model.addVar(lb=Qg_l[i], ub=Qg_u[i], vtype=GRB.CONTINUOUS, name='"Qg{0}"'.format(i))

    # Add system level constraints
    Aeq_p = hstack([Ct - Cf, zeros((nb, nl)), -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, zeros((nb, ng))])
    beq_p = bus[:, PD] / baseMVA
    # Add constraints for each sub system
    Aeq_q = hstack([zeros((nb, nl)), Ct - Cf, -diag(Ct * Branch_X) * Ct, zeros((nb, nb)), zeros((nb, ng)), Cg])
    beq_q = bus[:, QD] / baseMVA
    Aeq_KVL = hstack([-2 * diags(Branch_R), -2 * diags(Branch_X),
                      diags(turn_to_power(Branch_R, 2)) + diags(turn_to_power(Branch_X, 2)), Cf.T - Ct.T,
                      zeros((nl, 2 * ng))])
    beq_KVL = zeros(nl)

    Aeq = vstack([Aeq_p, Aeq_q, Aeq_KVL])
    Aeq = Aeq.todense()
    beq = concatenate([beq_p, beq_q, beq_KVL])
    neq = len(beq)
    nx = 3 * nl + nb + 2 * ng

    for i in range(neq):
        expr = 0
        for j in range(nx):
            expr += x[j] * Aeq[i, j]
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

    # for i in range(nl):
    #   model.addConstr(x[i]*x[i] + x[i+nl]*x[i+nl] <= x[i+2*nl]*x[f[i]+3*nl], name='"rc{0}"'.format(i))

    obj = 0
    for i in range(ng):
        obj += gencost[i, 4] * x[i + 3 * nl + nb] * x[i + 3 * nl + nb] * baseMVA * baseMVA + gencost[i, 5] * x[
            i + 3 * nl + nb] * baseMVA + gencost[i, 6]

    model.setObjective(obj)
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
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

    return xx, obj


def turn_to_power(list, power=1):
    return [number ** power for number in list]


if __name__ == "__main__":
    mpc = case33.case33()  # Default test case

    (xx, obj) = run(mpc)

    print(xx)
