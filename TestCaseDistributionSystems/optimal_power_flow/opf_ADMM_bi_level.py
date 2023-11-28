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
# 6 Feb 2018: The concept of observatory
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
    # Formulate the centralized optimization problem according to the information provided by area
    # Generate the initial value
    # 1) For the x-update
    Pi_x0 = [0] * nb
    Qi_x0 = [0] * nb
    Ii_x0 = [1] * nb
    Vi_x0 = [0] * nb
    pi_x0 = [0] * nb
    qi_x0 = [0] * nb
    Pg0 = [0] * nb
    Qg0 = [0] * nb
    # 2) For the y-update
    Pi_y0 = [0] * nb
    Qi_y0 = [0] * nb
    Ii_y0 = [0] * nb
    Vi_y0 = [1] * nb
    pi_y0 = [0] * nb
    qi_y0 = [0] * nb
    Pij_y0 = [0] * nl
    Qij_y0 = [0] * nl
    Iij_y0 = [0] * nl
    Vij_y0 = [0] * nl
    # 3) The multiplier part
    mu_Pi = [0] * nb
    mu_Qi = [0] * nb
    mu_Ii = [0] * nb
    mu_Vi = [0] * nb
    mu_pi = [0] * nb
    mu_qi = [0] * nb
    mu_Pij = [0] * nl
    mu_Qij = [0] * nl
    mu_Iij = [0] * nl
    mu_Vij = [0] * nl
    f = f.tolist()
    t = t.tolist()
    for i in range(nl):
        f[i] = int(f[i])
        t[i] = int(t[i])
    for i in range(nl):
        Pij_y0[i] = Pi_x0[t[i]]
        Qij_y0[i] = Qi_x0[t[i]]
        Iij_y0[i] = Ii_x0[t[i]]
        Vij_y0[i] = Vi_x0[f[i]]
    Gap = 1000
    Gap_index = []
    k = 0
    kmax = 10000
    ru = 1000
    half_ru = ru / 2
    # The iteration
    while k <= kmax and Gap > 0.001:
        # Y variables
        # Part 1), self observation
        modelY = Model("Yupdate")
        Pi_y = {}
        Qi_y = {}
        Ii_y = {}
        Vi_y = {}
        pi_y = {}
        qi_y = {}
        # Part 2), to the ancestor
        Pij_y = {}
        Qij_y = {}
        Iij_y = {}
        # Part 3), to the children. The definition is in accordance with the sequence of lines
        Vij_y = {}  # For the given branch
        for i in range(nb):  # The iteration from each bus
            Pi_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Pi_y{0}".format(i))
            Qi_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Qi_y{0}".format(i))
            Ii_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Ii_y{0}".format(i))
            Vi_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Vi_y{0}".format(i))
            pi_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi_y{0}".format(i))
            qi_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi_y{0}".format(i))

        for i in range(nl):  # The information stored in the observatory
            Pij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Pij_y{0}".format(i))
            Qij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Qij_y{0}".format(i))
            Iij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Iij_y{0}".format(i))
            Vij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Vij_y{0}".format(i))

        for i in range(nb):
            # Add constrain for the observation of each bus
            # 1)Constrain for KCL equations
            # 2)Constrain for KVL equations
            if area[i]["TYPE"] == "ROOT":
                # Only KCL equation is required
                expr = 0
                for j in range(area[i]["nCi"]):
                    expr += Pij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * area[i]["BR_R_C"][j]
                modelY.addConstr(pi_y[i] + expr == 0)

                expr = 0
                for j in range(area[i]["nCi"]):
                    expr += Qij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * area[i]["BR_X_C"][j]
                modelY.addConstr(qi_y[i] + expr == 0)

            elif area[i]["TYPE"] == "LEAF":  # Only KCL equation is required
                modelY.addConstr(pi_y[i] - Pi_y[i] == 0)
                modelY.addConstr(qi_y[i] - Qi_y[i] == 0)
                modelY.addConstr(
                    Vij_y[area[i]["Abranch"]] - Vi_y[i] + 2 * area[i]["BR_R"] * Pi_y[i] + 2 * area[i]["BR_X"] * Qi_y[
                        i] -
                    Ii_y[i] * (area[i]["BR_R"] ** 2 + area[i]["BR_X"] ** 2) == 0)
            else:
                expr = 0
                for j in range(area[i]["nCi"]):
                    expr += Pij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * area[i]["BR_R_C"][j]
                modelY.addConstr(pi_y[i] - Pi_y[i] + expr == 0)

                expr = 0
                for j in range(area[i]["nCi"]):
                    expr += Qij_y[area[i]["Cbranch"][j]] - Iij_y[area[i]["Cbranch"][j]] * area[i]["BR_X_C"][j]
                modelY.addConstr(qi_y[i] - Qi_y[i] + expr == 0)
                modelY.addConstr(
                    Vij_y[area[i]["Abranch"]] - Vi_y[i] + 2 * area[i]["BR_R"] * Pi_y[i] + 2 * area[i]["BR_X"] * Qi_y[
                        i] -
                    Ii_y[i] * (area[i]["BR_R"] ** 2 + area[i]["BR_X"] ** 2) == 0)
        objY = 0
        for i in range(nb):
            objY += mu_Pi[i] * Pi_y[i] + half_ru * (Pi_y[i] - Pi_x0[i]) * (Pi_y[i] - Pi_x0[i]) + \
                    mu_Qi[i] * Qi_y[i] + half_ru * (Qi_y[i] - Qi_x0[i]) * (Qi_y[i] - Qi_x0[i]) + \
                    mu_Ii[i] * Ii_y[i] + half_ru * (Ii_y[i] - Ii_x0[i]) * (Ii_y[i] - Ii_x0[i]) + \
                    mu_Vi[i] * Vi_y[i] + half_ru * (Vi_y[i] - Vi_x0[i]) * (Vi_y[i] - Vi_x0[i]) + \
                    mu_pi[i] * pi_y[i] + half_ru * (pi_y[i] - pi_x0[i]) * (pi_y[i] - pi_x0[i]) + \
                    mu_qi[i] * qi_y[i] + half_ru * (qi_y[i] - qi_x0[i]) * (qi_y[i] - qi_x0[i])
        for i in range(nl):
            objY += mu_Pij[i] * Pij_y[i] + half_ru * (Pij_y[i] - Pi_x0[t[i]]) * (Pij_y[i] - Pi_x0[t[i]]) + \
                    mu_Qij[i] * Qij_y[i] + half_ru * (Qij_y[i] - Qi_x0[t[i]]) * (Qij_y[i] - Qi_x0[t[i]]) + \
                    mu_Iij[i] * Iij_y[i] + half_ru * (Iij_y[i] - Ii_x0[t[i]]) * (Iij_y[i] - Ii_x0[t[i]]) + \
                    mu_Vij[i] * Vij_y[i] + half_ru * (Vij_y[i] - Vi_x0[f[i]]) * (Vij_y[i] - Vi_x0[f[i]])

        modelY.setObjective(objY)
        modelY.Params.OutputFlag = 0
        modelY.Params.LogToConsole = 0
        modelY.Params.DisplayInterval = 1
        modelY.Params.LogFile = ""
        modelY.optimize()
        for i in range(nb):
            Pi_y0[i] = modelY.getVarByName("Pi_y{0}".format(i)).X
            Qi_y0[i] = modelY.getVarByName("Qi_y{0}".format(i)).X
            Ii_y0[i] = modelY.getVarByName("Ii_y{0}".format(i)).X
            Vi_y0[i] = modelY.getVarByName("Vi_y{0}".format(i)).X
            pi_y0[i] = modelY.getVarByName("pi_y{0}".format(i)).X
            qi_y0[i] = modelY.getVarByName("qi_y{0}".format(i)).X

        for i in range(nl):
            Pij_y0[i] = modelY.getVarByName("Pij_y{0}".format(i)).X
            Qij_y0[i] = modelY.getVarByName("Qij_y{0}".format(i)).X
            Iij_y0[i] = modelY.getVarByName("Iij_y{0}".format(i)).X
            Vij_y0[i] = modelY.getVarByName("Vij_y{0}".format(i)).X
        del modelY

        # The sub problems
        modelX = Model("Xupdate")
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

        objX = 0
        for i in range(nb):  # The iteration from each bus
            Pi_x[i] = modelX.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                    name="Pi_x{0}".format(i))
            Qi_x[i] = modelX.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                    name="Qi_x{0}".format(i))
            Ii_x[i] = modelX.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                    name="Ii_x{0}".format(i))
            Vi_x[i] = modelX.addVar(lb=area[i]["VMIN"], ub=area[i]["VMAX"], vtype=GRB.CONTINUOUS,
                                    name="Vi_x{0}".format(i))

            pi_x[i] = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi_x{0}".format(i))
            qi_x[i] = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi_x{0}".format(i))

            Pg[i] = modelX.addVar(lb=area[i]["PGMIN"], ub=area[i]["PGMAX"], vtype=GRB.CONTINUOUS,
                                  name="Pgi{0}".format(i))
            Qg[i] = modelX.addVar(lb=area[i]["QGMIN"], ub=area[i]["QGMAX"], vtype=GRB.CONTINUOUS,
                                  name="Qgi{0}".format(i))
            # For each branch, the following observation variables should be introduced
            # According to the sequence of line

        for i in range(nb):
            # Add constrain for each bus
            modelX.addConstr(Pg[i] - pi_x[i] == area[i]["PD"])
            modelX.addConstr(Qg[i] - qi_x[i] == area[i]["QD"])
            modelX.addConstr(Pi_x[i] * Pi_x[i] + Qi_x[i] * Qi_x[i] <= Ii_x[i] * Vi_x[i])

        for i in range(nb):
            # Update the objective function
            objX += area[i]["a"] * Pg[i] * Pg[i] + area[i]["b"] * Pg[i] + area[i]["c"] - \
                    mu_Pi[i] * Pi_x[i] + half_ru * (Pi_x[i] - Pi_y0[i]) * (Pi_x[i] - Pi_y0[i]) - \
                    mu_Qi[i] * Qi_x[i] + half_ru * (Qi_x[i] - Qi_y0[i]) * (Qi_x[i] - Qi_y0[i]) - \
                    mu_Ii[i] * Ii_x[i] + half_ru * (Ii_x[i] - Ii_y0[i]) * (Ii_x[i] - Ii_y0[i]) - \
                    mu_Vi[i] * Vi_x[i] + half_ru * (Vi_x[i] - Vi_y0[i]) * (Vi_x[i] - Vi_y0[i]) - \
                    mu_pi[i] * pi_x[i] + half_ru * (pi_x[i] - pi_y0[i]) * (pi_x[i] - pi_y0[i]) - \
                    mu_qi[i] * qi_x[i] + half_ru * (qi_x[i] - qi_y0[i]) * (qi_x[i] - qi_y0[i])

        for i in range(nl):
            objX += -mu_Pij[i] * Pi_x[t[i]] + half_ru * (Pi_x[t[i]] - Pij_y0[i]) * (Pi_x[t[i]] - Pij_y0[i]) \
                    - mu_Qij[i] * Qi_x[t[i]] + half_ru * (Qi_x[t[i]] - Qij_y0[i]) * (Qi_x[t[i]] - Qij_y0[i]) \
                    - mu_Iij[i] * Ii_x[t[i]] + half_ru * (Ii_x[t[i]] - Iij_y0[i]) * (Ii_x[t[i]] - Iij_y0[i]) \
                    - mu_Vij[i] * Vi_x[f[i]] + half_ru * (Vi_x[f[i]] - Vij_y0[i]) * (Vi_x[f[i]] - Vij_y0[i])

        modelX.setObjective(objX)
        modelX.Params.OutputFlag = 0
        modelX.Params.LogToConsole = 0
        modelX.Params.DisplayInterval = 1
        modelX.Params.LogFile = ""
        modelX.optimize()
        for i in range(nb):
            Pi_x0[i] = modelX.getVarByName("Pi_x{0}".format(i)).X
            Qi_x0[i] = modelX.getVarByName("Qi_x{0}".format(i)).X
            Ii_x0[i] = modelX.getVarByName("Ii_x{0}".format(i)).X
            Vi_x0[i] = modelX.getVarByName("Vi_x{0}".format(i)).X
            pi_x0[i] = modelX.getVarByName("pi_x{0}".format(i)).X
            qi_x0[i] = modelX.getVarByName("qi_x{0}".format(i)).X
            Pg0[i] = modelX.getVarByName("Pgi{0}".format(i)).X
            Qg0[i] = modelX.getVarByName("Qgi{0}".format(i)).X
        del modelX

        # Update mutiplier
        for i in range(nb):
            mu_Pi[i] += ru * (Pi_y0[i] - Pi_x0[i])
            mu_Qi[i] += ru * (Qi_y0[i] - Qi_x0[i])
            mu_Ii[i] += ru * (Ii_y0[i] - Ii_x0[i])
            mu_Vi[i] += ru * (Vi_y0[i] - Vi_x0[i])
            mu_pi[i] += ru * (pi_y0[i] - pi_x0[i])
            mu_qi[i] += ru * (qi_y0[i] - qi_x0[i])
        for i in range(nl):
            mu_Pij[i] += ru * (Pij_y0[i] - Pi_x0[t[i]])
            mu_Qij[i] += ru * (Qij_y0[i] - Qi_x0[t[i]])
            mu_Iij[i] += ru * (Iij_y0[i] - Ii_x0[t[i]])
            mu_Vij[i] += ru * (Vij_y0[i] - Vi_x0[f[i]])

        # Update residual
        residual = 0
        for i in range(nb):
            residual += abs(Pi_y0[i] - Pi_x0[i])
            residual += abs(Qi_y0[i] - Qi_x0[i])
            residual += abs(Ii_y0[i] - Ii_x0[i])
            residual += abs(Vi_y0[i] - Vi_x0[i])
            residual += abs(pi_y0[i] - pi_x0[i])
            residual += abs(qi_y0[i] - qi_x0[i])
        for i in range(nl):
            residual += abs(Pij_y0[i] - Pi_x0[t[i]])
            residual += abs(Qij_y0[i] - Qi_x0[t[i]])
            residual += abs(Iij_y0[i] - Ii_x0[t[i]])
            residual += abs(Vij_y0[i] - Vi_x0[f[i]])

        obj = 0
        for i in range(nb):
            obj += area[i]["a"] * Pg0[i] * Pg0[i] + area[i]["b"] * Pg0[i] + area[i]["c"]
        Gap = residual
        k = k + 1
        print(k)
        print(Gap)
        print(obj)
    obj = 0
    for i in range(nb):
        obj += area[i]["a"] * Pg0[i] * Pg0[i] + area[i]["b"] * Pg0[i] + area[i]["c"]

    primal_residual = []
    for i in range(nb):
        primal_residual.append(Pi_x0[i] * Pi_x0[i] + Qi_x0[i] * Qi_x0[i] - Ii_x0[i] * Vi_x0[i])

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
            temp["BR_R_C"] = []
            temp["BR_X_C"] = []
            for j in range(nChildren):
                temp["Cbranch"].append(int(ChildrenBranch[0][j]))
                temp["Ci"].append(int(branch_t[temp["Cbranch"][j]]))  # The children bus
            temp["nCi"] = nChildren
            temp["BR_R_C"] = Branch_R[ChildrenBranch].tolist()
            temp["BR_X_C"] = Branch_X[ChildrenBranch].tolist()
        else:
            temp["Cbranch"] = []
            temp["Ci"] = []
            temp["nCi"] = 0
            temp["BR_R_C"] = []
            temp["BR_X_C"] = []

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
