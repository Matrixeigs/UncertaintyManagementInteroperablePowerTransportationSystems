"""
ADMM based distributed optimal power flow
The power flow modelling is based on the branch power flow
Notes：
1） The concept of observatory has been introduced and verified
2) Based on multi-block ADMM
3) Convergence verified
4) Simulation gap found! (related to the parameter of ru)

"""

from distribution_system_optimization.test_cases import case33
from distribution_system_optimization.test_cases import case123, case69
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, where, inf
from scipy.sparse import csr_matrix as sparse
# Data format
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int
from math import sqrt
from copy import deepcopy
from numpy import array
import matplotlib.pyplot as plt


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
    f = branch[:, F_BUS].astype(int)  ## list of "from" buses
    t = branch[:, T_BUS].astype(int)  ## list of "to" buses

    # Modify the bus information
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
    # Initialize algorithm for each sub area
    # 1) For each area, for self observation
    # x:=[Pg,Qg,pi,qi,Pi,Qi,Vi,Ii]
    # y:=[pi,qi,Pi,Qi,Vi,Ii,Pij,Qij,Vij,Iij]
    f = f.tolist()
    t = t.tolist()
    for i in range(nb):
        area[i]["PG"] = (area[i]["PGMAX"] + area[i]["PGMIN"]) / 2
        area[i]["QG"] = (area[i]["QGMIN"] + area[i]["QGMAX"]) / 2
        # Observation of x
        area[i]["pi"] = area[i]["PG"] - area[i]["PD"]
        area[i]["qi"] = area[i]["QG"] - area[i]["QD"]
        area[i]["Pi"] = area[i]["pi"]
        area[i]["Qi"] = area[i]["qi"]
        area[i]["Vi"] = (area[i]["VMIN"] + area[i]["VMAX"]) / 2
        if area[i]["TYPE"] != "ROOT":
            area[i]["Ii"] = (area[i]["Pi"] ** 2 + area[i]["Qi"] ** 2) / area[i]["Vi"]
        # The self observation
        area[i]["pi_y"] = area[i]["pi"]
        area[i]["qi_y"] = area[i]["qi"]
        if area[i]["TYPE"] != "ROOT":
            area[i]["Vi_y"] = area[i]["Vi"]
            area[i]["Ii_y"] = area[i]["Ii"]
            area[i]["Pi_y"] = area[i]["Pi"]
            area[i]["Qi_y"] = area[i]["Qi"]
        # The multipliers
        area[i]["mu_pi"] = area[i]["pi"] - area[i]["pi_y"]
        area[i]["mu_qi"] = area[i]["qi"] - area[i]["qi_y"]
        if area[i]["TYPE"] != "ROOT":
            area[i]["mu_Vi"] = area[i]["Vi"] - area[i]["Vi_y"]
            area[i]["mu_Ii"] = area[i]["Ii"] - area[i]["Ii_y"]
            area[i]["mu_Pi"] = area[i]["Pi"] - area[i]["Pi_y"]
            area[i]["mu_Qi"] = area[i]["Qi"] - area[i]["Qi_y"]
        area[i]["COST"] = 0
        # Spread the information to the observatory

    observatory = []
    # Store the voltage of parent bus and children power flow information
    # 1)The ancestor bus voltage information is stored in the observotory
    # 2)The children bus power and current information is stored in the observotory
    for i in range(nl):
        temp = {}
        temp["Vij_x"] = area[int(f[i])]["Vi"]
        temp["Pij_x"] = area[int(t[i])]["Pi"]
        temp["Qij_x"] = area[int(t[i])]["Qi"]
        temp["Iij_x"] = area[int(t[i])]["Ii"]
        temp["Vij_y"] = area[int(f[i])]["Vi"]
        temp["Pij_y"] = area[int(t[i])]["Pi"]
        temp["Qij_y"] = area[int(t[i])]["Qi"]
        temp["Iij_y"] = area[int(t[i])]["Ii"]
        temp["mu_Vij"] = 0
        temp["mu_Pij"] = 0
        temp["mu_Qij"] = 0
        temp["mu_Iij"] = 0
        observatory.append(temp)
    # Begin the iteration,
    Gap = 1000
    Gap_index = []
    Dual_gap_index = []
    Obj_index = []
    k = 0
    kmax = 10000
    ru = 100
    # The iteration
    mu = 100
    t = 1.5
    while k <= kmax and Gap > 0.0001:
        observatory0 = deepcopy(observatory)
        area0 = deepcopy(area)
        for i in range(nb):
            (area, observatory) = sub_problem(area, observatory, i, ru / 2)

        # # multiplier update
        # for i in range(nb):
        #     area[i]["mu_pi"] += ru * (area[i]["pi"] - area[i]["pi_y"])
        #     area[i]["mu_qi"] += ru * (area[i]["qi"] - area[i]["qi_y"])
        #     if area[i]["TYPE"] != "ROOT":
        #         area[i]["mu_Vi"] += ru * (area[i]["Vi"] - area[i]["Vi_y"])
        #         area[i]["mu_Ii"] += ru * (area[i]["Ii"] - area[i]["Ii_y"])
        #         area[i]["mu_Pi"] += ru * (area[i]["Pi"] - area[i]["Pi_y"])
        #         area[i]["mu_Qi"] += ru * (area[i]["Qi"] - area[i]["Qi_y"])
        # for i in range(nl):
        #     observatory[i]["mu_Vij"] += ru * (observatory[i]["Vij_x"] - observatory[i]["Vij_y"])
        #     observatory[i]["mu_Pij"] += ru * (observatory[i]["Pij_x"] - observatory[i]["Pij_y"])
        #     observatory[i]["mu_Qij"] += ru * (observatory[i]["Qij_x"] - observatory[i]["Qij_y"])
        #     observatory[i]["mu_Iij"] += ru * (observatory[i]["Iij_x"] - observatory[i]["Iij_y"])

        # Calculate the gap
        gap = 0
        for i in range(nb):
            gap += (area[i]["pi"] - area[i]["pi_y"]) ** 2
            gap += (area[i]["qi"] - area[i]["qi_y"]) ** 2
            if area[i]["TYPE"] != "ROOT":
                gap += (area[i]["Vi"] - area[i]["Vi_y"]) ** 2
                gap += (area[i]["Ii"] - area[i]["Ii_y"]) ** 2
                gap += (area[i]["Pi"] - area[i]["Pi_y"]) ** 2
                gap += (area[i]["Qi"] - area[i]["Qi_y"]) ** 2
        for i in range(nl):
            gap += (observatory[i]["Vij_x"] - observatory[i]["Vij_y"]) ** 2
            gap += (observatory[i]["Pij_x"] - observatory[i]["Pij_y"]) ** 2
            gap += (observatory[i]["Qij_x"] - observatory[i]["Qij_y"]) ** 2
            gap += (observatory[i]["Iij_x"] - observatory[i]["Iij_y"]) ** 2

        dual_gap = 0
        for i in range(nb):
            dual_gap += (area0[i]["pi_y"] - area[i]["pi_y"]) ** 2
            dual_gap += (area0[i]["qi_y"] - area[i]["qi_y"]) ** 2
            if area[i]["TYPE"] != "ROOT":
                dual_gap += (area0[i]["Vi_y"] - area[i]["Vi_y"]) ** 2
                dual_gap += (area0[i]["Ii_y"] - area[i]["Ii_y"]) ** 2
                dual_gap += (area0[i]["Pi_y"] - area[i]["Pi_y"]) ** 2
                dual_gap += (area0[i]["Qi_y"] - area[i]["Qi_y"]) ** 2
        for i in range(nl):
            dual_gap += (observatory0[i]["Vij_y"] - observatory[i]["Vij_y"]) ** 2
            dual_gap += (observatory0[i]["Pij_y"] - observatory[i]["Pij_y"]) ** 2
            dual_gap += (observatory0[i]["Qij_y"] - observatory[i]["Qij_y"]) ** 2
            dual_gap += (observatory0[i]["Iij_y"] - observatory[i]["Iij_y"]) ** 2

        # if sqrt(dual_gap) * mu < Gap:
        #     ru = ru * t
        # if Gap * mu < sqrt(dual_gap):
        #     ru = ru / t

        Gap = sqrt(gap)
        Gap_index.append(Gap)
        Dual_gap_index.append(sqrt(dual_gap))
        obj = 0
        for i in range(nb):
            obj += area[i]["COST"]
        k = k + 1
        if k % 10 == 0:
            print(k)
            print(Gap)
            print(obj)
            print(sqrt(dual_gap))
            # print(ru)
    # compute the objective function
    obj = 0
    for i in range(nb):
        obj += area[i]["COST"]
    Gap_index = array(Gap_index)
    Dual_gap_index = array(Dual_gap_index)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(Gap_index)
    plt.ylabel('Primal residual')

    plt.subplot(212)
    plt.plot(Dual_gap_index)
    plt.ylabel('Dual residual')

    plt.show()
    return obj, area


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
            # try:
            #     temp["Abranch"] = int(AncestorBranch[0])
            # except:
            #     temp["Abranch"] = int(AncestorBranch[0][0])
            temp["Abranch"] = int(AncestorBranch[0])
            print(i)
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


def sub_problem(area, observatory, index, half_ru):
    """
    Sub-problem optimization for each area, where the area is defined one bus and together with the
    :param Index: Target area
    :param area: Area connection information
    :param observatory: Observatory information
    :param index: Target area
    :param Ru: penalty factor in the objective function
    :return: Updated area and observatory information
    """
    modelX = Model("sub_opf_x")  # Sub-optimal power flow x update
    modelY = Model("sub_opf_y")  # Sub-optimal power flow y update

    if area[index]["TYPE"] == "ROOT":  # Only needs to meet the KCL equation
        # xi = [Pgi,Qgi,pi_x,qi_x,Vi_x]
        # zi= [pi_z,qi_z,Pj_i_z(j in children set of i),Qj_i_z,Ij_i_z]
        # The following types of constraints should be considered
        # 1) pi_x = pi_z
        # 2) qi_x = qi_z
        # 3) Vi_x = V_A_j_z j in children set of i

        # 4）Pj_x = Pj_i_z j in children set of i, active power from j to i
        # 5）Qj_x = Qj_i_z j in children set of i
        # 6）Ij_x = Ij_i_z j in children set of i
        # constraint 4)-6) are managed by the children buses in the y-update problem

        # Information exchange
        # In the z update, receive Pj_x, Qj_x, Ij_x and associate Lagrange multiper from children set
        # In the x update, receive V_A_j_z from children set

        # Step 1: construct the X update problem
        X = ["PG", "QG", "pi", "qi", "Vi"]
        Y = ["pi_y", "qi_y"]
        # Obtain information from the children bus, for voltage
        # 1) Update information from the observatory (connected via Cbranch)
        # 1.1) Update the center information
        Vi0 = [0] * area[index]["nCi"]
        for i in range(area[index]["nCi"]):
            Vi0[i] = observatory[area[index]["Cbranch"][i]]["Vij_y"]
        pi0 = area[index]["pi_y"]
        qi0 = area[index]["qi_y"]
        # 1.2) Update the multiplier information
        mu_Vi = [0] * area[index]["nCi"]
        for i in range(area[index]["nCi"]):
            mu_Vi[i] = observatory[area[index]["Cbranch"][i]]["mu_Vij"]
        mu_pi = area[index]["mu_pi"]
        mu_qi = area[index]["mu_qi"]
        # 1.3) Variable announcement
        Pg = modelX.addVar(lb=area[index]["PGMIN"], ub=area[index]["PGMAX"], vtype=GRB.CONTINUOUS, name="PG")
        Qg = modelX.addVar(lb=area[index]["QGMIN"], ub=area[index]["QGMAX"], vtype=GRB.CONTINUOUS, name="QG")
        pi = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi")
        qi = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi")
        Vi = modelX.addVar(lb=area[index]["VMIN"], ub=area[index]["VMAX"], vtype=GRB.CONTINUOUS, name="Vi")
        # 1.4) Formulate constraints (power balance equations)
        modelX.addConstr(Pg - pi == area[index]["PD"])
        modelX.addConstr(Qg - qi == area[index]["QD"])
        # 1.5) Formulate objective function
        objX = area[index]["a"] * Pg * Pg + area[index]["b"] * Pg + area[index]["c"] + \
               mu_pi * pi + half_ru * (pi - pi0) * (pi - pi0) + \
               mu_qi * qi + half_ru * (qi - qi0) * (qi - qi0)
        for i in range(area[index]["nCi"]):
            objX += mu_Vi[i] * Vi + half_ru * (Vi - Vi0[i]) * (Vi - Vi0[i])
        # 1.6) Solve the problem
        modelX.setObjective(objX)
        modelX.Params.OutputFlag = 0
        modelX.Params.LogToConsole = 0
        modelX.Params.DisplayInterval = 1
        modelX.Params.LogFile = ""
        modelX.optimize()
        # 1.7) Update the solution
        for i in X:
            area[index][i] = modelX.getVarByName(i).X
        # 1.8) Update the observatory information
        for i in range(area[index]["nCi"]):
            observatory[area[index]["Cbranch"][i]]["Vij_x"] = area[index]["Vi"]
        # 1.9) Update objective function
        area[index]["COST"] = area[index]["a"] * area[index]["PG"] ** 2 + area[index]["b"] * area[index]["PG"] + \
                              area[index]["c"]
        del modelX
        # Step 2: construct the Y update problem
        # 1) Update information from the observatory (connected via Cbranch)
        # 1.1) Update the center information
        pi_y0 = area[index]["pi"]
        qi_y0 = area[index]["qi"]
        Pij_y0 = []
        Qij_y0 = []
        Iij_y0 = []
        for i in range(area[index]["nCi"]):
            Pij_y0.append(observatory[area[index]["Cbranch"][i]]["Pij_x"])
            Qij_y0.append(observatory[area[index]["Cbranch"][i]]["Qij_x"])
            Iij_y0.append(observatory[area[index]["Cbranch"][i]]["Iij_x"])
        # 1.2) Update the multiplier information
        mu_pi_y = area[index]["mu_pi"]
        mu_qi_y = area[index]["mu_qi"]
        mu_Pij = []
        mu_Qij = []
        mu_Iij = []
        for i in range(area[index]["nCi"]):
            mu_Pij.append(observatory[area[index]["Cbranch"][i]]["mu_Pij"])
            mu_Qij.append(observatory[area[index]["Cbranch"][i]]["mu_Qij"])
            mu_Iij.append(observatory[area[index]["Cbranch"][i]]["mu_Iij"])
        # 1.3) Variable announcement
        pi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi_y")
        qi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi_y")
        Pij_y = {}
        Qij_y = {}
        Iij_y = {}
        for i in range(area[index]["nCi"]):
            Pij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Pij_y{0}".format(i))
            Qij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Qij_y{0}".format(i))
            Iij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Iij_y{0}".format(i))
        # 1.4) Formulate constraints (KCL functions)
        expr = 0
        for i in range(area[index]["nCi"]):
            expr += Pij_y[i] - Iij_y[i] * area[index]["BR_R_C"][i]
        modelY.addConstr(pi_y + expr == 0)
        expr = 0
        for i in range(area[index]["nCi"]):
            expr += Qij_y[i] - Iij_y[i] * area[index]["BR_X_C"][i]
        modelY.addConstr(qi_y + expr == 0)
        # 1.5) Formulate objective function
        objY = -mu_pi_y * pi_y + half_ru * (pi_y - pi_y0) * (pi_y - pi_y0) \
               - mu_qi_y * qi_y + half_ru * (qi_y - qi_y0) * (qi_y - qi_y0)
        for i in range(area[index]["nCi"]):
            objY += -mu_Pij[i] * Pij_y[i] + half_ru * (Pij_y[i] - Pij_y0[i]) * (Pij_y[i] - Pij_y0[i]) \
                    - mu_Qij[i] * Qij_y[i] + half_ru * (Qij_y[i] - Qij_y0[i]) * (Qij_y[i] - Qij_y0[i]) \
                    - mu_Iij[i] * Iij_y[i] + half_ru * (Iij_y[i] - Iij_y0[i]) * (Iij_y[i] - Iij_y0[i])
        # 1.6) Solve the problem
        modelY.setObjective(objY)
        modelY.Params.OutputFlag = 0
        modelY.Params.LogToConsole = 0
        modelY.Params.DisplayInterval = 1
        modelY.Params.LogFile = ""
        modelY.optimize()
        # 1.7) Update the solution
        for i in Y:
            area[index][i] = modelY.getVarByName(i).X
        # 1.8) Update the observatory information
        # 1.8.1) Update ancestor information(power and current inforamtion, y)
        for i in range(area[index]["nCi"]):
            observatory[area[index]["Cbranch"][i]]["Pij_y"] = modelY.getVarByName("Pij_y{0}".format(i)).X
            observatory[area[index]["Cbranch"][i]]["Qij_y"] = modelY.getVarByName("Qij_y{0}".format(i)).X
            observatory[area[index]["Cbranch"][i]]["Iij_y"] = modelY.getVarByName("Iij_y{0}".format(i)).X
            observatory[area[index]["Cbranch"][i]]["mu_Pij"] += half_ru * (
                    observatory[area[index]["Cbranch"][i]]["Pij_x"] - observatory[area[index]["Cbranch"][i]][
                "Pij_y"]) * 2
            observatory[area[index]["Cbranch"][i]]["mu_Qij"] += half_ru * (
                    observatory[area[index]["Cbranch"][i]]["Qij_x"] - observatory[area[index]["Cbranch"][i]][
                "Qij_y"]) * 2
            observatory[area[index]["Cbranch"][i]]["mu_Iij"] += half_ru * (
                    observatory[area[index]["Cbranch"][i]]["Iij_x"] - observatory[area[index]["Cbranch"][i]][
                "Iij_y"]) * 2
        del modelY
        # Step 3: local multipiler update
        area[index]["mu_pi"] += half_ru * (area[index]["pi"] - area[index]["pi_y"]) * 2
        area[index]["mu_qi"] += half_ru * (area[index]["qi"] - area[index]["qi_y"]) * 2
    elif area[index]["TYPE"] == "LEAF":  # Only needs to meet the KVL equation
        # xi = [Pgi,Qgi,pi_x,qi_x,Vi_x,Ii_x,Pi_x,Qi_x]# Pi_x represent the power from i to its ancestor
        # zi = [qi_z,pi_z,Vi_z,Ii_z,Pi_z,Qi_z,V_A_i_z]#
        # The following types of constraints should be considered
        # 1) pi_x = pi_z
        # 2) qi_x = qi_z
        # 3) Vi_x = Vi_z
        # 4) Ii_x = Ii_z
        # 5) Pi_x = Pi_z
        # 6) Qi_x = Qi_z
        # 7) Pi_x = Pi_j_z j is the ancestor bus of i
        # 8) Qi_x = Qi_j_z j is the ancestor bus of i
        # 9) Ii_x = Ii_j_z j is the ancestor bus of i
        # 10) Vj_x = V_A_i_z  j is the ancestor bus of i

        # Information exchange
        # In the z update, receive Vj_x and associate Lagrange multiper from ancestor bus
        # In the x update, receive Pi_j_z, Qi_j_z and Ii_j_z from ancestor bus
        X = ["PG", "QG", "pi", "qi", "Pi", "Qi", "Ii", "Vi"]
        Y = ["pi_y", "qi_y", "Pi_y", "Qi_y", "Ii_y", "Vi_y"]
        # Step 1: construct the X update problem
        # 1) Update information from the observatory (connected via Abranch)
        # 1.1) Update the center information
        Pi0 = [observatory[area[index]["Abranch"]]["Pij_y"], area[index]["Pi_y"]]
        Qi0 = [observatory[area[index]["Abranch"]]["Qij_y"], area[index]["Qi_y"]]
        Ii0 = [observatory[area[index]["Abranch"]]["Iij_y"], area[index]["Ii_y"]]
        Vi0 = area[index]["Vi_y"]
        pi0 = area[index]["pi_y"]
        qi0 = area[index]["qi_y"]
        # 1.2) Update the multiplier information
        mu_Pi = [observatory[area[index]["Abranch"]]["mu_Pij"], area[index]["mu_Pi"]]
        mu_Qi = [observatory[area[index]["Abranch"]]["mu_Qij"], area[index]["mu_Qi"]]
        mu_Ii = [observatory[area[index]["Abranch"]]["mu_Iij"], area[index]["mu_Ii"]]
        mu_Vi = area[index]["mu_Vi"]
        mu_pi = area[index]["mu_pi"]
        mu_qi = area[index]["mu_qi"]
        # 1.3) Variable announcement
        Pg = modelX.addVar(lb=area[index]["PGMIN"], ub=area[index]["PGMAX"], vtype=GRB.CONTINUOUS, name="PG")
        Qg = modelX.addVar(lb=area[index]["QGMIN"], ub=area[index]["QGMAX"], vtype=GRB.CONTINUOUS, name="QG")
        Pi = modelX.addVar(lb=-area[index]["SMAX"], ub=area[index]["SMAX"], vtype=GRB.CONTINUOUS, name="Pi")
        Qi = modelX.addVar(lb=-area[index]["SMAX"], ub=area[index]["SMAX"], vtype=GRB.CONTINUOUS, name="Qi")
        Ii = modelX.addVar(lb=-area[index]["SMAX"], ub=area[index]["SMAX"], vtype=GRB.CONTINUOUS, name="Ii")
        Vi = modelX.addVar(lb=area[index]["VMIN"], ub=area[index]["VMAX"], vtype=GRB.CONTINUOUS, name="Vi")
        pi = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi")
        qi = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi")
        # 1.4) Formulate constraints(nodal power balance equation)
        modelX.addConstr(Pg - pi == area[index]["PD"])
        modelX.addConstr(Qg - qi == area[index]["QD"])
        modelX.addConstr(Pi * Pi + Qi * Qi <= Vi * Ii)
        # 1.5) Formulate objective function
        objX = area[index]["a"] * Pg * Pg + area[index]["b"] * Pg + area[index]["c"] \
               + mu_Vi * Vi + half_ru * (Vi - Vi0) * (Vi - Vi0) \
               + mu_pi * pi + half_ru * (pi - pi0) * (pi - pi0) \
               + mu_qi * qi + half_ru * (qi - qi0) * (qi - qi0)

        for i in range(len(Pi0)):
            objX += mu_Pi[i] * Pi + half_ru * (Pi - Pi0[i]) * (Pi - Pi0[i]) \
                    + mu_Qi[i] * Qi + half_ru * (Qi - Qi0[i]) * (Qi - Qi0[i]) \
                    + mu_Ii[i] * Ii + half_ru * (Ii - Ii0[i]) * (Ii - Ii0[i])

        # 1.6) Solve the problem
        modelX.setObjective(objX)
        modelX.Params.OutputFlag = 0
        modelX.Params.LogToConsole = 0
        modelX.Params.DisplayInterval = 1
        modelX.Params.LogFile = ""
        modelX.optimize()
        # 1.7) Update the solution
        for i in X:
            area[index][i] = modelX.getVarByName(i).X
        del modelX
        # 1.8) Update the observatory information
        observatory[area[index]["Abranch"]]["Pij_x"] = area[index]["Pi"]
        observatory[area[index]["Abranch"]]["Qij_x"] = area[index]["Qi"]
        observatory[area[index]["Abranch"]]["Iij_x"] = area[index]["Ii"]
        # 1.9) Update objective function
        area[index]["COST"] = area[index]["a"] * area[index]["PG"] ** 2 + area[index]["b"] * area[index]["PG"] + \
                              area[index]["c"]

        # Step 2: construct the Y update problem
        # 1) Update information from the observatory (connected via Abranch)
        # 1.1) Update the center information
        pi_y0 = area[index]["pi"]
        qi_y0 = area[index]["qi"]
        Pi_y0 = area[index]["Pi"]
        Qi_y0 = area[index]["Qi"]
        Ii_y0 = area[index]["Ii"]
        Vi_y0 = area[index]["Vi"]
        Vij_y0 = observatory[area[index]["Abranch"]]["Vij_x"]
        # 1.2) Update the multiplier information
        mu_Pi_y = area[index]["mu_Pi"]
        mu_Qi_y = area[index]["mu_Qi"]
        mu_Ii_y = area[index]["mu_Ii"]
        mu_Vi_y = area[index]["mu_Vi"]
        mu_pi_y = area[index]["mu_pi"]
        mu_qi_y = area[index]["mu_qi"]
        mu_Vij_y = observatory[area[index]["Abranch"]]["mu_Vij"]
        # 1.3) Variable announcement
        Pi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Pi_y")
        Qi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Qi_y")
        Ii_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Ii_y")
        Vi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Vi_y")
        pi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi_y")
        qi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi_y")
        Vij_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Vij_y")
        # 1.4) Formulate constraints
        # 1.4.1) KCL equation
        modelY.addConstr(pi_y - Pi_y == 0)
        modelY.addConstr(qi_y - Qi_y == 0)
        # 1.4.2) KVL equation
        modelY.addConstr(Vij_y - Vi_y + 2 * area[index]["BR_R"] * Pi_y + 2 * area[index]["BR_X"] * Qi_y - Ii_y * (
                area[index]["BR_R"] ** 2 + area[index]["BR_X"] ** 2) == 0)
        # 1.5) Formulate objective function
        objY = -mu_Pi_y * Pi_y + half_ru * (Pi_y - Pi_y0) * (Pi_y - Pi_y0) \
               - mu_Qi_y * Qi_y + half_ru * (Qi_y - Qi_y0) * (Qi_y - Qi_y0) \
               - mu_Ii_y * Ii_y + half_ru * (Ii_y - Ii_y0) * (Ii_y - Ii_y0) \
               - mu_Vi_y * Vi_y + half_ru * (Vi_y - Vi_y0) * (Vi_y - Vi_y0) \
               - mu_pi_y * pi_y + half_ru * (pi_y - pi_y0) * (pi_y - pi_y0) \
               - mu_qi_y * qi_y + half_ru * (qi_y - qi_y0) * (qi_y - qi_y0) \
               - mu_Vij_y * Vij_y + half_ru * (Vij_y - Vij_y0) * (Vij_y - Vij_y0)
        # 1.6) Solve the problem
        modelY.setObjective(objY)
        modelY.Params.OutputFlag = 0
        modelY.Params.LogToConsole = 0
        modelY.Params.DisplayInterval = 1
        modelY.Params.LogFile = ""
        modelY.optimize()
        # 1.7) Update the solution
        for i in Y:
            area[index][i] = modelY.getVarByName(i).X
        # 1.8) Update the observatory information
        # 1.8.1) Update ancestor information(voltage, y)
        observatory[area[index]["Abranch"]]["Vij_y"] = modelY.getVarByName("Vij_y").X
        observatory[area[index]["Abranch"]]["mu_Vij"] += half_ru * (
                observatory[area[index]["Abranch"]]["Vij_x"] - observatory[area[index]["Abranch"]]["Vij_y"]) * 2
        del modelY
        # Step 3: local multipiler update
        area[index]["mu_pi"] += half_ru * (area[index]["pi"] - area[index]["pi_y"]) * 2
        area[index]["mu_qi"] += half_ru * (area[index]["qi"] - area[index]["qi_y"]) * 2
        area[index]["mu_Vi"] += half_ru * (area[index]["Vi"] - area[index]["Vi_y"]) * 2
        area[index]["mu_Ii"] += half_ru * (area[index]["Ii"] - area[index]["Ii_y"]) * 2
        area[index]["mu_Pi"] += half_ru * (area[index]["Pi"] - area[index]["Pi_y"]) * 2
        area[index]["mu_Qi"] += half_ru * (area[index]["Qi"] - area[index]["Qi_y"]) * 2

    else:  # Only needs to meet the KVL equation
        # xi = [Pgi,Qgi,pi_x,pi_x,Vi_x,Ii_x,Pi_x,Qi_x]# Pi_x represent the power from i to its ancestor
        # zi = [pi_z,pi_z,Vi_z,Ii_z,Pi_z,Qi_z,V_A_i_z,Pj_i_z(j in children set of i),Qj_i_z,Ij_i_z]#
        # The following types of constraints should be considered
        # 1) pi_x = pi_z
        # 2) qi_x = qi_z
        # 3) Vi_x = Vi_z
        # 4) Ii_x = Ii_z
        # 5) Pi_x = Pi_z
        # 6) Qi_x = Qi_z
        # 7) Pi_x = Pi_j_z j is the ancestor bus of i
        # 8) Qi_x = Qi_j_z j is the ancestor bus of i
        # 9) Ii_x = Ii_j_z j is the ancestor bus of i
        # 10) Vi_x = V_A_j_z  j in the children bus of i

        # 11) Vj_x = V_A_i_z j is the ancestor bus of i
        # 11）Pj_x = Pj_i_z j in children set of i
        # 12）Qj_x = Qj_i_z j in children set of i
        # 13）Ij_x = Ij_i_z j in children set of i

        # Information exchange
        # In the z update, receive Pj_x, Qj_x, Ij_x and associate Lagrange multiper from children set; Vj_x and associate Lagrange multiper from ancestor bus
        # In the x update, receive Pi_j_z, Qi_j_z and Ii_j_z from ancestor bus; and V_A_j_z from children set. In the x update, no multiper is required.
        X = ["PG", "QG", "pi", "qi", "Pi", "Qi", "Ii", "Vi"]
        Y = ["pi_y", "qi_y", "Pi_y", "Qi_y", "Ii_y", "Vi_y"]

        # Step 1: construct the X update problem
        # 1) Update information from the observatory (connected via Abranch and Cbranches)
        # 1.1) Update the center information
        # 1.1.1) From Abranch obtain the power and current information
        Pi0 = [observatory[area[index]["Abranch"]]["Pij_y"], area[index]["Pi_y"]]
        Qi0 = [observatory[area[index]["Abranch"]]["Qij_y"], area[index]["Qi_y"]]
        Ii0 = [observatory[area[index]["Abranch"]]["Iij_y"], area[index]["Ii_y"]]
        # 1.1.2) From Cbranch obtain the voltage information
        Vi0 = [0] * (area[index]["nCi"] + 1)
        for i in range(area[index]["nCi"]):
            Vi0[i] = observatory[area[index]["Cbranch"][i]]["Vij_y"]
        Vi0[area[index]["nCi"]] = area[index]["Vi_y"]
        pi0 = area[index]["pi_y"]
        qi0 = area[index]["qi_y"]
        # 1.2) Update the multiplier information
        # 1.2.1) From Abranch obtain the power and current information
        mu_Pi = [observatory[area[index]["Abranch"]]["mu_Pij"], area[index]["mu_Pi"]]
        mu_Qi = [observatory[area[index]["Abranch"]]["mu_Qij"], area[index]["mu_Qi"]]
        mu_Ii = [observatory[area[index]["Abranch"]]["mu_Iij"], area[index]["mu_Ii"]]
        mu_pi = area[index]["mu_pi"]
        mu_qi = area[index]["mu_qi"]
        # 1.2.2) From Cbranch obtain the voltage information
        mu_Vi = [0] * (area[index]["nCi"] + 1)
        for i in range(area[index]["nCi"]):
            mu_Vi[i] = observatory[area[index]["Cbranch"][i]]["mu_Vij"]
        mu_Vi[area[index]["nCi"]] = area[index]["mu_Vi"]
        # 1.3) Variable announcement
        Pg = modelX.addVar(lb=area[index]["PGMIN"], ub=area[index]["PGMAX"], vtype=GRB.CONTINUOUS, name="PG")
        Qg = modelX.addVar(lb=area[index]["QGMIN"], ub=area[index]["QGMAX"], vtype=GRB.CONTINUOUS, name="QG")
        Pi = modelX.addVar(lb=-area[index]["SMAX"], ub=area[index]["SMAX"], vtype=GRB.CONTINUOUS, name="Pi")
        Qi = modelX.addVar(lb=-area[index]["SMAX"], ub=area[index]["SMAX"], vtype=GRB.CONTINUOUS, name="Qi")
        Ii = modelX.addVar(lb=-area[index]["SMAX"], ub=area[index]["SMAX"], vtype=GRB.CONTINUOUS, name="Ii")
        Vi = modelX.addVar(lb=area[index]["VMIN"], ub=area[index]["VMAX"], vtype=GRB.CONTINUOUS, name="Vi")
        pi = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi")
        qi = modelX.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi")
        # 1.4) Formulate constraints
        # 1.4.1) Power balance equation
        modelX.addConstr(Pg - pi == area[index]["PD"])
        modelX.addConstr(Qg - qi == area[index]["QD"])
        # 1.4.2) Power definition
        modelX.addConstr(Pi * Pi + Qi * Qi <= Vi * Ii)
        # 1.5) Formulate objective function
        objX = area[index]["a"] * Pg * Pg + area[index]["b"] * Pg + area[index]["c"] + \
               mu_pi * pi + half_ru * (pi - pi0) * (pi - pi0) + \
               mu_qi * qi + half_ru * (qi - qi0) * (qi - qi0)
        for i in range(area[index]["nCi"]):
            objX += mu_Vi[i] * Vi + half_ru * (Vi - Vi0[i]) * (Vi - Vi0[i])

        for i in range(len(Pi0)):
            objX += mu_Pi[i] * Pi + half_ru * (Pi - Pi0[i]) * (Pi - Pi0[i]) \
                    + mu_Qi[i] * Qi + half_ru * (Qi - Qi0[i]) * (Qi - Qi0[i]) \
                    + mu_Ii[i] * Ii + half_ru * (Ii - Ii0[i]) * (Ii - Ii0[i])
        # 1.6) Solve the problem
        modelX.setObjective(objX)
        modelX.Params.OutputFlag = 0
        modelX.Params.LogToConsole = 0
        modelX.Params.DisplayInterval = 1
        modelX.Params.LogFile = ""
        modelX.optimize()
        # 1.7) Update the solution
        for i in X:
            area[index][i] = modelX.getVarByName(i).X
        del modelX
        # 1.8) Update the observatory information
        observatory[area[index]["Abranch"]]["Pij_x"] = area[index]["Pi"]
        observatory[area[index]["Abranch"]]["Qij_x"] = area[index]["Qi"]
        observatory[area[index]["Abranch"]]["Iij_x"] = area[index]["Ii"]
        for i in range(area[index]["nCi"]):
            observatory[area[index]["Cbranch"][i]]["Vij_x"] = area[index]["Vi"]
        # 1.9) Update objective function
        area[index]["COST"] = area[index]["a"] * area[index]["PG"] ** 2 + area[index]["b"] * area[index]["PG"] + \
                              area[index]["c"]
        # Step 2: construct the Y update problem
        # 1) Update information from the observatory (connected via Abranch and Cbranches)
        # 1.1) Update the center information
        # 1.1.1) From Cbranch obtain the power and current information
        pi_y0 = area[index]["pi"]
        qi_y0 = area[index]["qi"]
        Pi_y0 = area[index]["Pi"]
        Qi_y0 = area[index]["Qi"]
        Ii_y0 = area[index]["Ii"]
        Vi_y0 = area[index]["Vi"]
        # 1.1.1) From Abranch obtain the voltage information
        Vij_y0 = observatory[area[index]["Abranch"]]["Vij_x"]
        # 1.1.2) From Cbranch obtain the power and current information
        Pij_y0 = []
        Qij_y0 = []
        Iij_y0 = []
        for i in range(area[index]["nCi"]):
            Pij_y0.append(observatory[area[index]["Cbranch"][i]]["Pij_x"])
            Qij_y0.append(observatory[area[index]["Cbranch"][i]]["Qij_x"])
            Iij_y0.append(observatory[area[index]["Cbranch"][i]]["Iij_x"])
        # 1.2) Update the multiplier information
        mu_Pi_y = area[index]["mu_Pi"]
        mu_Qi_y = area[index]["mu_Qi"]
        mu_Ii_y = area[index]["mu_Ii"]
        mu_Vi_y = area[index]["mu_Vi"]
        mu_pi_y = area[index]["mu_pi"]
        mu_qi_y = area[index]["mu_qi"]
        # 1.2.1) From Abranch obtain the voltage information
        mu_Vij_y = observatory[area[index]["Abranch"]]["mu_Vij"]
        # 1.2.2) From Abranch obtain the power and current information
        mu_Pij = []
        mu_Qij = []
        mu_Iij = []
        for i in range(area[index]["nCi"]):
            mu_Pij.append(observatory[area[index]["Cbranch"][i]]["mu_Pij"])
            mu_Qij.append(observatory[area[index]["Cbranch"][i]]["mu_Qij"])
            mu_Iij.append(observatory[area[index]["Cbranch"][i]]["mu_Iij"])
        # 1.3) Variable announcement
        Pi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Pi_y")
        Qi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Qi_y")
        Ii_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Ii_y")
        Vi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Vi_y")
        pi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="pi_y")
        qi_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="qi_y")
        Vij_y = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Vij_y")
        Pij_y = {}
        Qij_y = {}
        Iij_y = {}
        for i in range(area[index]["nCi"]):
            Pij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Pij_y{0}".format(i))
            Qij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Qij_y{0}".format(i))
            Iij_y[i] = modelY.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name="Iij_y{0}".format(i))
        # 1.4) Formulate constraints
        # 1.4.1) KCL equation
        expr = 0
        for i in range(area[index]["nCi"]):
            expr += Pij_y[i] - Iij_y[i] * area[index]["BR_R_C"][i]
        modelY.addConstr(pi_y + expr == Pi_y)
        expr = 0
        for i in range(area[index]["nCi"]):
            expr += Qij_y[i] - Iij_y[i] * area[index]["BR_X_C"][i]
        modelY.addConstr(qi_y + expr == Qi_y)
        # 1.4.2) KVL equation
        modelY.addConstr(Vij_y - Vi_y + 2 * area[index]["BR_R"] * Pi_y + 2 * area[index]["BR_X"] * Qi_y - Ii_y * (
                area[index]["BR_R"] ** 2 + area[index]["BR_X"] ** 2) == 0)
        # 1.5) Formulate objective function
        objY = -mu_Pi_y * Pi_y + half_ru * (Pi_y - Pi_y0) * (Pi_y - Pi_y0) \
               - mu_Qi_y * Qi_y + half_ru * (Qi_y - Qi_y0) * (Qi_y - Qi_y0) \
               - mu_Ii_y * Ii_y + half_ru * (Ii_y - Ii_y0) * (Ii_y - Ii_y0) \
               - mu_Vi_y * Vi_y + half_ru * (Vi_y - Vi_y0) * (Vi_y - Vi_y0) \
               - mu_pi_y * pi_y + half_ru * (pi_y - pi_y0) * (pi_y - pi_y0) \
               - mu_qi_y * qi_y + half_ru * (qi_y - qi_y0) * (qi_y - qi_y0) \
               - mu_Vij_y * Vij_y + half_ru * (Vij_y - Vij_y0) * (Vij_y - Vij_y0)
        for i in range(area[index]["nCi"]):
            objY += -mu_Pij[i] * Pij_y[i] + half_ru * (Pij_y[i] - Pij_y0[i]) * (Pij_y[i] - Pij_y0[i]) + \
                    -mu_Qij[i] * Qij_y[i] + half_ru * (Qij_y[i] - Qij_y0[i]) * (Qij_y[i] - Qij_y0[i]) + \
                    -mu_Iij[i] * Iij_y[i] + half_ru * (Iij_y[i] - Iij_y0[i]) * (Iij_y[i] - Iij_y0[i])
        # 1.6) Solve the problem
        modelY.setObjective(objY)
        modelY.Params.OutputFlag = 0
        modelY.Params.LogToConsole = 0
        modelY.Params.DisplayInterval = 1
        modelY.Params.LogFile = ""
        modelY.optimize()
        # 1.7) Update the solution
        for i in Y:
            area[index][i] = modelY.getVarByName(i).X
        # 1.8) Update the observatory information
        # 1.8.1) Update ancestor information(voltage,y)
        observatory[area[index]["Abranch"]]["Vij_y"] = modelY.getVarByName("Vij_y").X
        observatory[area[index]["Abranch"]]["mu_Vij"] += half_ru * (
                observatory[area[index]["Abranch"]]["Vij_x"] - observatory[area[index]["Abranch"]]["Vij_y"]) * 2
        # 1.8.2) Update children information(power and current, y)
        for i in range(area[index]["nCi"]):
            observatory[area[index]["Cbranch"][i]]["Pij_y"] = modelY.getVarByName("Pij_y{0}".format(i)).X
            observatory[area[index]["Cbranch"][i]]["Qij_y"] = modelY.getVarByName("Qij_y{0}".format(i)).X
            observatory[area[index]["Cbranch"][i]]["Iij_y"] = modelY.getVarByName("Iij_y{0}".format(i)).X
            observatory[area[index]["Cbranch"][i]]["mu_Pij"] += half_ru * (
                    observatory[area[index]["Cbranch"][i]]["Pij_x"] - observatory[area[index]["Cbranch"][i]][
                "Pij_y"]) * 2
            observatory[area[index]["Cbranch"][i]]["mu_Qij"] += half_ru * (
                    observatory[area[index]["Cbranch"][i]]["Qij_x"] - observatory[area[index]["Cbranch"][i]][
                "Qij_y"]) * 2
            observatory[area[index]["Cbranch"][i]]["mu_Iij"] += half_ru * (
                    observatory[area[index]["Cbranch"][i]]["Iij_x"] - observatory[area[index]["Cbranch"][i]][
                "Iij_y"]) * 2
        del modelY
        # Step 3: local multipiler update
        area[index]["mu_pi"] += half_ru * (area[index]["pi"] - area[index]["pi_y"]) * 2
        area[index]["mu_qi"] += half_ru * (area[index]["qi"] - area[index]["qi_y"]) * 2
        area[index]["mu_Vi"] += half_ru * (area[index]["Vi"] - area[index]["Vi_y"]) * 2
        area[index]["mu_Ii"] += half_ru * (area[index]["Ii"] - area[index]["Ii_y"]) * 2
        area[index]["mu_Pi"] += half_ru * (area[index]["Pi"] - area[index]["Pi_y"]) * 2
        area[index]["mu_Qi"] += half_ru * (area[index]["Qi"] - area[index]["Qi_y"]) * 2

    return area, observatory


if __name__ == "__main__":
    from pypower import runopf
    mpc = case69.case69()  # Default test case

    # mpc = case123.case123()  # Default test case
    # mpc = case33.case33()  # Default test case
    (obj, residual) = run(mpc)

    # result = runopf.runopf(mpc)

    # gap = 100 * (result["f"] - obj) / obj

    print()
