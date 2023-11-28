"""
Scenario reduction algorithm for two-stage stochastic programmings
The fast forward selection algorithm is used.

References:
    [1]https://edoc.hu-berlin.de/bitstream/handle/18452/3285/8.pdf?sequence=1
    [2]http://ftp.gamsworld.org/presentations/present_IEEE03.pdf
    [3] https://arxiv.org/pdf/1701.04072.pdf
Considering the second stage optimization problem is linear programming, the distance function is refined to
c(ξ, ˜ξ) := max{1, kξkp−1, k˜ξkp−1}kξ − ˜ξk (p = 2, which is sufficient for right hand side uncertainties)

"""

from numpy import array, zeros, argmin, random, arange, linalg, ones, inf, delete, where, append
from Solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp
from scipy.sparse import lil_matrix, vstack
import numpy as np
class ScenarioReduction():
    def __init__(self):
        self.name = "Scenario reduction"

    def run(self, scenario, weight, n_reduced, power):
        """

        :param scenario: A fan scenario tree, when more stage are considered, some merge operation can be implemented
        :param weight: Weight of each scenario
        :param n_reduced: Number of scenarios needs to be reduced
        :param power: The power in the distance calculation
        :return:
        """
        n_scenario = scenario.shape[0]  # number of original scenarios
        c = zeros((n_scenario, n_scenario))
        # Calculate the c matrix
        for i in range(n_scenario):
            for j in range(n_scenario):
                c[i, j] = linalg.norm((scenario[i, :] - scenario[j, :]), 2)
                c[i, j] = max([1, linalg.norm(scenario[i, :], power - 1), linalg.norm(scenario[j, :], power - 1)]) * \
                          c[i, j]

        J = arange(n_scenario)  # The original index range
        J_reduced = array([])
        # Implement the iteration
        for n in range(n_reduced):  # find the minimal distance
            print("The reduction is in process {0}".format(n))
            c_n = inf * ones(n_scenario)
            c_n[J] = 0
            for u in J:
                # Delete the i-th distance
                J_temp = delete(J, where(J == u))
                for k in J_temp:
                    c_k_j = delete(c[int(k)], J_temp)
                    c_n[int(u)] += weight[int(k)] * min(c_k_j)
            u_i = argmin(c_n)
            J_reduced = append(J_reduced, u_i)
            J = delete(J, where(J == u_i))
        # Optimal redistribution
        p_s = weight.copy()
        p_s[J_reduced.astype(int)] = 0

        for i in J_reduced:
            c_temp = c[int(i), :]
            c_temp[J_reduced.astype(int)] = inf
            index = argmin(c_temp)
            p_s[index] += weight[int(i)]

        scenario_reduced = scenario[J.astype(int), :]
        weight_reduced = p_s[J.astype(int)]

        return scenario_reduced, weight_reduced

class ScenarioReductionDiscrete():
    """ Based on the problem 21 in Ref[3], a mixed-integer linear programming method is formulated"""
    def __init__(self):
        self.name = "Scenario reduction under discrete scenario"
    def run(self, scenario, weight, n_reduced, power):
        """
        :param scenario: A fan scenario tree, when more stage are considered, some merge operation can be implemented
        :param weight: Weight of each scenario
        :param n_reduced: Number of scenarios needs to be reduced
        :param power: The power in the distance calculation
        :return:
        """
        n_scenario = scenario.shape[0]  # number of original scenarios
        D = zeros((n_scenario, n_scenario))
        # Calculate the c matrix
        for i in range(n_scenario):
            for j in range(i+1):
                D[i, j] = linalg.norm((scenario[i, :] - scenario[j, :]), 2)
                D[i, j] = max([1, linalg.norm(scenario[i, :], power - 1), linalg.norm(scenario[j, :], power - 1)]) * \
                          D[i, j]
        D = D+D.transpose() # To reduce the computational cost
        # The matrix is sorted row wise
        nx = n_scenario*n_scenario + n_scenario
        lb = zeros((nx,1))
        ub = ones((nx,1))
        vtypes = ["c"]*n_scenario*n_scenario+["b"]*n_scenario
        c = zeros((nx,1))
        c[0:n_scenario*n_scenario] = D.reshape(n_scenario*n_scenario,1)/n_scenario
        Aeq = lil_matrix((n_scenario, nx),dtype=int)
        beq = zeros((n_scenario, 1))
        for s in range(n_scenario):
            Aeq[s,s*n_scenario:(s+1)*n_scenario] = 1
            beq[s,0] = 1

        Aeq_temp = lil_matrix((1,nx),dtype=float)
        beq_temp = ones((1,1))*(n_scenario-n_reduced)
        Aeq_temp[0,n_scenario*n_scenario:] = 1
        Aeq = vstack([Aeq,Aeq_temp]).tolil()
        beq = np.vstack([beq,beq_temp])

        A = lil_matrix((n_scenario*n_scenario+1, nx),dtype=int)
        b = zeros((n_scenario*n_scenario+1, 1))
        for i in range(n_scenario):
            for j in range(n_scenario):
                A[i*n_scenario+j, i*n_scenario+j] = 1
                A[i*n_scenario+j, n_scenario*n_scenario+j] = -1

        sol,obj,success = milp(c,Aeq=Aeq,beq=beq,A=A,b=b,xmin=lb,xmax=ub,vtypes=vtypes)
        index = sol[n_scenario*n_scenario:]
        ws = array(sol[0:n_scenario*n_scenario]).reshape(n_scenario,n_scenario)
        ws_ori = ws.sum(axis=0)
        j = 0
        index_set = []
        for i in range(n_scenario):
            if index[i]>0:
                index_set.append(j)
            j += 1
        scenario_reduced = zeros((n_scenario - n_reduced, scenario.shape[1]))
        weight_reduced = zeros((n_scenario-n_reduced,1))
        for i in range(n_scenario-n_reduced):
            scenario_reduced[i,:] = scenario[index_set[i],:]
            weight_reduced[i] = ws_ori[index_set[i]]/n_scenario



        return scenario_reduced, weight_reduced



if __name__ == "__main__":
    n_scenario = 100
    scenario = random.random((n_scenario, 10))
    weight = ones(n_scenario) / n_scenario
    n_reduced = int(n_scenario*0.95)
    power = 2
    scenario_reduction = ScenarioReductionDiscrete()

    (scenario_reduced, weight_reduced) = scenario_reduction.run(scenario=scenario, weight=weight, n_reduced=n_reduced,
                                                                power=power)

    print(weight_reduced)
