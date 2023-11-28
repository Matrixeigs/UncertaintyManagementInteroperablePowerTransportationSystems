"""
ADMM based distributed optimal power flow
The power flow modelling is based on the branch power flow
The purpose is to introduce slack variables to formulate a decentralized optimization problem.
The centralized model (24a)
References:
    [1]Peng, Qiuyu, and Steven H. Low. "Distributed optimal power flow algorithm for radial networks, I: Balanced single phase case." IEEE Transactions on Smart Grid (2016).
# Each bus is equivalent
# xi=[Pi,Qi,li,vi,pi,qi,pgi,pdi]
# yij=[Pi,Qi,li,vi,pi,qi,vCi,PiAi,QiAi,liAi]
# for each y, its size might be changed
# In total, the size of y equals to 5*nb+3*nl(current and power)+nl(ancestor bus voltage)
# 31 Jan 2018: The concept of observatory
"""

from distribution_system_optimization.test_cases import case33
from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, where, inf
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags
# Data format
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int


def run(mpc):
    """
    Gurobi based optimal power flow modelling and solution
    :param mpc: The input case of optimal power flow
    :return: obtained solution
    """
    mpc = ext2int(mpc)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

    nb = shape(mpc['bus'])[0]  # number of buses
    nl = shape(mpc['branch'])[0]  # number of branches
    ng = shape(mpc['gen'])[0]  # number of dispatchable injections
    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    i = range(nl)  ## double set of row indices
    # Connection matrix
    Cf = sparse((ones(nl), (i, f)), (nl, nb))
    Ct = sparse((ones(nl), (i, t)), (nl, nb))
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    # Connection matrix
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    Branch_R = branch[:, BR_R]
    Branch_X = branch[:, BR_X]
    Slmax = branch[:, RATE_A] / baseMVA
    gen[:, PMAX] = gen[:, PMAX] / baseMVA
    gen[:, PMIN] = gen[:, PMIN] / baseMVA
    gen[:, QMAX] = gen[:, QMAX] / baseMVA
    gen[:, QMIN] = gen[:, QMIN] / baseMVA
    gencost[:, 4] = gencost[:, 4] * baseMVA * baseMVA
    gencost[:, 5] = gencost[:, 5] * baseMVA
    bus[:, PD] = bus[:, PD] / baseMVA
    bus[:, QD] = bus[:, QD] / baseMVA
    area = ancestor_children_generation(f, t, nb, Branch_R, Branch_X, Slmax, gen, bus, gencost, baseMVA)
    M = inf
    # Formulate the centralized optimization problem according to the information provided by area
    model = Model("OPF")

    # Define the decision variables, compact set
    # X variables
    Pi_x = {}
    Qi_x = {}
    Ii_x = {}
    Vi_x = {}
    pi_x = {}
    qi_x = {}
    Pg = {}
    Qg = {}

    # Y variables
    # Part 1), self observation
    Pii_y = {}
    Qii_y = {}
    Iii_y = {}
    Vii_y = {}
    pii_y = {}
    qii_y = {}
    # Part 2), to the ancestor
    Pij_y = {}
    Qij_y = {}
    Iij_y = {}
    # Part 3), to the children. The definition is in accordance with the sequence of lines
    Vij_y = {}  # For the given branch

    obj = 0
    j = 0
    for i in range(nb):  # The iteration from each bus
        Pi_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                               name="Pi_x{0}".format(i))
        Qi_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                               name="Qi_x{0}".format(i))
        Ii_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                               name="Ii_x{0}".format(i))
        Vi_x[i] = model.addVar(lb=area[i]["VMIN"], ub=area[i]["VMAX"], vtype=GRB.CONTINUOUS,
                               name="Vi_x{0}".format(i))

        pi_x[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="pi_x{0}".format(i))
        qi_x[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="qi_x{0}".format(i))

        Pg[i] = model.addVar(lb=area[i]["PGMIN"], ub=area[i]["PGMAX"], vtype=GRB.CONTINUOUS, name="Pgi{0}".format(i))
        Qg[i] = model.addVar(lb=area[i]["QGMIN"], ub=area[i]["QGMAX"], vtype=GRB.CONTINUOUS, name="Qgi{0}".format(i))

        Pii_y[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Pii_y{0}".format(i))
        Qii_y[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Qii_y{0}".format(i))
        Iii_y[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Iii_y{0}".format(i))
        Vii_y[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Vii_y{0}".format(i))
        pii_y[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="pii_y{0}".format(i))
        qii_y[i] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="qii_y{0}".format(i))
        # For each branch, the following observation variables should be introduced
        # According to the sequence of lines

        if area[i]["TYPE"] != "ROOT":  # If this bus is the root bus
            Pij_y[j] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Pij_y{0}".format(j))
            Qij_y[j] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Qij_y{0}".format(j))
            Iij_y[j] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Iij_y{0}".format(j))
            Vij_y[j] = model.addVar(lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="Vij_y{0}".format(j))
            j += 1

    for i in range(nb):
        # Add constrain for each bus
        model.addConstr(Pg[i] - pi_x[i] == area[i]["PD"])
        model.addConstr(Qg[i] - qi_x[i] == area[i]["QD"])
        model.addConstr(Pi_x[i] * Pi_x[i] + Qi_x[i] * Qi_x[i] <= Ii_x[i] * Vi_x[i])

        # Update the objective function
        obj += area[i]["a"] * Pg[i] * Pg[i] + area[i]["b"] * Pg[i] + area[i]["c"]
        # Add constrain for the observation of each bus
        # 1)Constrain for KCL equations
        # 2)Constrain for KVL equations
        if area[i]["TYPE"] == "ROOT":
            # Only KCL equation is required
            expr = 0
            for j in range(len(area[i]["Ci"])):
                expr += Pij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * Branch_R[area[i]["Cbranch"][j]]
            model.addConstr(pii_y[i] + expr == 0)

            expr = 0
            for j in range(len(area[i]["Ci"])):
                expr += Qij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * Branch_X[area[i]["Cbranch"][j]]
            model.addConstr(qii_y[i] + expr == 0)

        elif area[i]["TYPE"] == "LEAF":  # Only KCL equation is required
            model.addConstr(pii_y[i] - Pii_y[i] == 0)
            model.addConstr(qii_y[i] - Qii_y[i] == 0)
            model.addConstr(
                Vij_y[area[i]["Abranch"]] - Vii_y[i] + 2 * area[i]["BR_R"] * Pii_y[i] + 2 * area[i]["BR_X"] * Qii_y[i] -
                Iii_y[i] * (area[i]["BR_R"] ** 2 + area[i]["BR_X"] ** 2) == 0)
        else:
            expr = 0
            for j in range(len(area[i]["Ci"])):
                expr += Pij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * Branch_R[area[i]["Cbranch"][j]]
            model.addConstr(pii_y[i] - Pii_y[i] + expr == 0)

            expr = 0
            for j in range(len(area[i]["Ci"])):
                expr += Qij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * Branch_X[area[i]["Cbranch"][j]]
            model.addConstr(qii_y[i] - Qii_y[i] + expr == 0)
            model.addConstr(
                Vij_y[area[i]["Abranch"]] - Vii_y[i] + 2 * area[i]["BR_R"] * Pii_y[i] + 2 * area[i]["BR_X"] * Qii_y[i] -
                Iii_y[i] * (area[i]["BR_R"] ** 2 + area[i]["BR_X"] ** 2) == 0)

        # Formulate consensus constraints
        # Add constraints
        # The introduction of Xii is to formulate the closed form of the solution
        model.addConstr(Pii_y[i] == Pi_x[i])
        model.addConstr(Qii_y[i] == Qi_x[i])
        model.addConstr(Vii_y[i] == Vi_x[i])
        model.addConstr(Iii_y[i] == Ii_x[i])
        model.addConstr(pii_y[i] == pi_x[i])
        model.addConstr(qii_y[i] == qi_x[i])
        # For each branch
    # for i in range(nl): # which stands for the observatory for each line; The observatory constraints
    #     model.addConstr(Vij_y[i] == Vi_x[f[i]])
    #     model.addConstr(Pij_y[i] == Pi_x[t[i]])
    #     model.addConstr(Qij_y[i] == Qi_x[t[i]])
    #     model.addConstr(Iij_y[i] == Ii_x[t[i]])
    # from the perspective of nodes
    for i in range(nb):
        if area[i]["nChildren"] != 0:
            for j in range(area[i]["nChildren"]):
                model.addConstr(Vi_x[i] == Vij_y[area[i]["Cbranch"][j]])
        if area[i]["TYPE"] != "ROOT":
            model.addConstr(Pi_x[i] == Pij_y[area[i]["Abranch"]])
            model.addConstr(Qi_x[i] == Qij_y[area[i]["Abranch"]])
            model.addConstr(Ii_x[i] == Iij_y[area[i]["Abranch"]])
    model.setObjective(obj)
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.DisplayInterval = 1
    model.optimize()

    Pi = []
    Qi = []
    Ii = []
    Vi = []
    pi = []
    qi = []
    pg = []
    qg = []

    for i in range(nb):
        Pi.append(model.getVarByName("Pi_x{0}".format(i)).X)
        Qi.append(model.getVarByName("Pi_x{0}".format(i)).X)
        Ii.append(model.getVarByName("Ii_x{0}".format(i)).X)
        Vi.append(model.getVarByName("Vi_x{0}".format(i)).X)
        pi.append(model.getVarByName("pi_x{0}".format(i)).X)
        qi.append(model.getVarByName("qi_x{0}".format(i)).X)
        pg.append(model.getVarByName("Pgi{0}".format(i)).X)
        qg.append(model.getVarByName("Qgi{0}".format(i)).X)

    obj = obj.getValue()

    primal_residual = []

    for i in range(nb):
        primal_residual.append(Pi[i] * Pi[i] + Qi[i] * Qi[i] - Ii[i] * Vi[i])

    return obj, primal_residual


def turn_to_power(list, power=1):
    return [number ** power for number in list]


def ancestor_children_generation(branch_f, branch_t, nb, Branch_R, Branch_X, SMAX, gen, bus, gencost, baseMVA):
    """
    Ancestor and children information for each node, together with information within each area
    :param branch_f:
    :param branch_t:
    :param index: Bus index
    :param Branch_R: Branch resistance
    :param Branch_X: Branch reactance
    :param SMAX: Current limitation within each area
    :param gen: Generation information
    :param bus: Bus information
    :return: Area, ancestor bus, children buses, line among buses, load, generations and line information
    """
    Area = []
    for i in range(nb):
        temp = {}
        temp["Index"] = i
        if i in branch_t:
            AncestorBus = branch_f[where(branch_t == i)]
            temp["Ai"] = int(AncestorBus[0])  # For each bus, there exits only one ancestor bus, as one connected tree
            AncestorBranch = where(branch_t == i)
            temp["Abranch"] = int(AncestorBranch[0])
            temp["BR_R"] = Branch_R[temp["Abranch"]]
            temp["BR_X"] = Branch_X[temp["Abranch"]]
            temp["SMAX"] = SMAX[temp["Abranch"]]
            if i in branch_f:
                temp["TYPE"] = "RELAY"
            else:
                temp["TYPE"] = "LEAF"
        else:
            temp["Ai"] = []
            temp["Abranch"] = []
            temp["BR_R"] = 0
            temp["BR_X"] = 0
            temp["SMAX"] = 0
            temp["TYPE"] = "ROOT"

        if i in branch_f:
            ChildrenBranch = where(branch_f == i)
            nChildren = len(ChildrenBranch[0])
            temp["Ci"] = []
            temp["Cbranch"] = []
            for j in range(nChildren):
                temp["Cbranch"].append(int(ChildrenBranch[0][j]))
                temp["Ci"].append(int(branch_t[temp["Cbranch"][j]]))  # The children bus
            temp["nChildren"] = nChildren
        else:
            temp["Cbranch"] = []
            temp["Ci"] = []
            temp["nChildren"] = 0

        # Update the node information
        if i in gen[:, GEN_BUS]:
            temp["PGMAX"] = gen[where(gen[:, GEN_BUS] == i), PMAX][0][0]
            temp["PGMIN"] = gen[where(gen[:, GEN_BUS] == i), PMIN][0][0]
            temp["QGMAX"] = gen[where(gen[:, GEN_BUS] == i), QMAX][0][0]
            temp["QGMIN"] = gen[where(gen[:, GEN_BUS] == i), QMIN][0][0]
            if temp["PGMIN"] > temp["PGMAX"]:
                t = temp["PGMIN"]
                temp["PGMIN"] = temp["PGMAX"]
                temp["PGMAX"] = t
            if temp["QGMIN"] > temp["QGMAX"]:
                t = temp["QGMIN"]
                temp["QGMIN"] = temp["QGMAX"]
                temp["QGMAX"] = t
            temp["a"] = gencost[where(gen[:, GEN_BUS] == i), 4][0][0]
            temp["b"] = gencost[where(gen[:, GEN_BUS] == i), 5][0][0]
            temp["c"] = gencost[where(gen[:, GEN_BUS] == i), 6][0][0]
        else:
            temp["PGMAX"] = 0
            temp["PGMIN"] = 0
            temp["QGMAX"] = 0
            temp["QGMIN"] = 0
            temp["a"] = 0
            temp["b"] = 0
            temp["c"] = 0
        temp["PD"] = bus[i, PD]
        temp["QD"] = bus[i, QD]
        temp["VMIN"] = bus[i, VMIN] ** 2
        temp["VMAX"] = bus[i, VMAX] ** 2

        Area.append(temp)

    return Area


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    (obj, residual) = run(mpc)

    result = runopf.runopf(case33.case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)
    print(residual)
