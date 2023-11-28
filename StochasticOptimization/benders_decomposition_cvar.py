"""
Benders decomposition method for two-stage stochastic optimization problems
    Minimize a function F(X) beginning, subject to
	optional linear and nonlinear constraints and variable bounds:
			min  c'*x + sum(p_s*Q_s(x)) + \ru CVAR_{alpha}[Q_s(x)]
			x
			s.t. A*x<=b,
			     Aeq*x==beq,   x \in [lb,ub]
			where Q_s(x)=min q_s'*ys
			             y
			             s.t. W_s*y > h_s-T_s*x # The standard format, in accordance with robust optimization
References:
    [1]Benders Decomposition for Solving Two-stage Stochastic Optimization Models
    https://www.ima.umn.edu/materials/2015-2016/ND8.1-12.16/25378/Luedtke-spalgs.pdf
    [2]http://www.iems.ucf.edu/qzheng/grpmbr/seminar/Yuping_Intro_to_BendersDecomp.pdf
    [3]http://artax.karlin.mff.cuni.cz/~branm1am/download/VK_Benders.pdf
    [4]http://www.optimization-online.org/DB_FILE/2010/03/2571.pdf

@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 27 March 2020
@version: 1.0

notes:
1) The data structure is based on the numpy and scipy
2) This algorithm should be extended for further version to solve the jointed chance constrained stochastic programming
3) In this test algorithm, Mosek is adpoted. https://www.mosek.com/
4) In the second stage optimization, the dual problem is solved, so that only the cplex problem is needed to solve the problem.
5) The multi-cuts version Benders decomposition is adopted.
6) Dual cuts methods are adopted
7) Conditional value at risk is integrated
"""
from Solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
from numpy import zeros, hstack, vstack, transpose, ones, inf, array, arange, delete
from copy import deepcopy
from Solvers.benders_solvers import linear_programming as lp_dual
from multiprocessing import Pool
import os


class BendersDecomposition():
    def __init__(self):
        self.name = "Benders decomposition using C&CG method"

    def main(self, c=None, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, vtype=None, ps=None, qs=None, Ws=None,
             hs=None, Ts=None, alpha=0.9, ru=0.1, M = 10 ** 12, eps = 10 ** 0):
        """
        The standard input format for Benders decomposition problem
        :param c: Cost parameter for the first stage optimization
        :param A: Inequality constraint matrix for the first stage optimization
        :param b: Inequality constraint parameters for the first stage optimization
        :param Aeq: Equality constraint matrix for the first stage optimization
        :param beq: Equality constraint parameters for the first stage optimization
        :param vtype: The type for the first stage optimization problems
        :param ps: Probability for the second stage optimization problem under scenario s
        :param qs: Cost parameters for the second stage optimization problem, a list of arrays
        :param Ws: Equality constraint parameters for the second stage optimization, a list of arrays
        :param hs: Equality constraint parameters for the second stage optimization
        :param Ts: Equality constraint matrix between the first stage and the second stage optimization
        :param alpha: Confidential level, this should be always be
        :param ru: Weight factor on CVAR
        :param eps: Convergence criteria
        :return: The obtained solution for the first stage optimization
        """
        # 1) Try to solve the first stage optimization problem
        model_first_stage = {"c": c,
                             "Aeq": Aeq,
                             "beq": beq,
                             "A": A,
                             "b": b,
                             "lb": lb,
                             "ub": ub,
                             "vtypes": vtype}

        sol_first_stage = BendersDecomposition.master_problem(self, model_first_stage)

        if sol_first_stage["status"] == 0:
            print("The master problem is infeasible!")
            return
        else:
            print("The master problem is feasible, the process continutes!")

        self.N = len(ps)  # The number of second stage decision variables

        self.nx_second_stage = Ws[0].shape[1]
        self.nx_first_stage = lb.shape[0]
        self.M = M
        model_second_stage = [0] * self.N

        for i in range(self.N):
            model_second_stage[i] = {"c": qs[i],
                                     "A": Ws[i],
                                     "hs": hs[i],
                                     "Ts": Ts[i],
                                     "lb": None,
                                     "ub": None}
        # 2) Reformulate the first stage optimization problem
        # 2.1) Estimate the boundary of the first stage optimization problem.
        # 2.2) Add additional variables to the first stage optimization problem
        # Using the multiple cuts version
        model_master = deepcopy(model_first_stage)
        model_master["c"] = vstack([model_first_stage["c"], ps, ps/(1-alpha)*ru,array(ru)])
        N_additional = self.N+self.N+1
        if model_master["Aeq"] is not None:
            model_master["Aeq"] = hstack([model_first_stage["Aeq"], zeros((model_first_stage["Aeq"].shape[0], N_additional))])
        if model_master["A"] is not None:
            model_master["A"] = hstack([model_first_stage["A"], zeros((model_first_stage["A"].shape[0], N_additional))])

        if model_master["lb"] is not None:
            model_master["lb"] = vstack([model_first_stage["lb"], -ones((self.N, 1)) * M, zeros((self.N,1)), zeros((1,1))])
        else:
            model_master["lb"] = vstack([-ones((self.N + self.nx_first_stage, 1)) * M, zeros((self.N,1)), zeros((1,1))])

        if model_master["ub"] is not None:
            model_master["ub"] = vstack([model_first_stage["ub"], ones((N_additional, 1)) * M])
        else:
            model_master["ub"] = ones((self.N + N_additional, 1)) * M

        if model_master["vtypes"] is not None:
            model_master["vtypes"] = model_first_stage["vtypes"] + ["c"] * N_additional
        else:
            model_master["vtypes"] = ["c"] * (self.nx_first_stage + N_additional)

        # 3) Reformulate the second stage optimization problem
        # 3.1) Formulate the dual problem for each problem under dual problems
        # The dual problem is solved
        x_first_stage = array(sol_first_stage["x"][0:self.nx_first_stage]).reshape(self.nx_first_stage, 1)
        model_second_stage = self.sub_problems_update(model_second_stage, x_first_stage)
        n_processors = os.cpu_count()
        with Pool(n_processors) as p:
            sol_second_stage = list(p.map(sub_problem_dual, model_second_stage))
        ## For Q(x,wi)<=eta
        A_cuts = zeros((self.N, self.nx_first_stage + N_additional))
        b_cuts = zeros((self.N, 1))
        for i in range(self.N):
            # Solve the dual problem
            A_cuts[i, 0:self.nx_first_stage] = -transpose(
                transpose(model_second_stage[i]["Ts"]).dot(sol_second_stage[i]["x"]))
            b_cuts[i, 0] = -transpose(sol_second_stage[i]["x"]).dot(model_second_stage[i]["hs"])
            if sol_second_stage[i]["status"] == 1:  # if the primal problem is feasible, add feasible cuts
                A_cuts[i, self.nx_first_stage + i] = -1
        ## For Q(x,wi)-CVAR<=eta, the chance constraints part
        A_cuts_add = zeros((self.N, self.nx_first_stage + N_additional))
        b_cuts_add = zeros((self.N, 1))
        for i in range(self.N):
            # Solve the dual problem
            if sol_second_stage[i]["status"] == 1:  # if the primal problem is feasible, add feasible cuts
                A_cuts_add[i, 0:self.nx_first_stage] = -transpose(
                transpose(model_second_stage[i]["Ts"]).dot(sol_second_stage[i]["x"]))
                b_cuts_add[i, 0] = -transpose(sol_second_stage[i]["x"]).dot(model_second_stage[i]["hs"])
                A_cuts_add[i, self.nx_first_stage + self.N + i] = -1 #vi
                A_cuts_add[i, self.nx_first_stage + self.N*2] = -1 #CVAR

        Upper = [inf]
        Lower = sol_first_stage["objvalue"]
        Gap = [self.gap_calculaiton(Upper[0], Lower)]
        iter_max = 10000
        iter = 0
        # 4) Begin the iteration
        while iter < iter_max:
            # Update the master problem
            if model_master["A"] is not None:
                model_master["A"] = vstack([model_master["A"], A_cuts,A_cuts_add])
            else:
                model_master["A"] = vstack([A_cuts,A_cuts_add])
            if model_master["b"] is not None:
                model_master["b"] = vstack([model_master["b"], b_cuts,b_cuts_add])
            else:
                model_master["b"] = vstack([b_cuts,b_cuts_add])

            # solve the master problem
            sol_first_stage = self.master_problem(model_master)
            Lower = max(sol_first_stage["objvalue"],Lower)

            # update the second stage solution
            x_first_stage = array(sol_first_stage["x"][0:self.nx_first_stage]).reshape(self.nx_first_stage, 1)

            model_second_stage = self.sub_problems_update(model_second_stage, x_first_stage)
            obj_value_second_stage = zeros((self.N, 1))

            with Pool(n_processors) as p:
                sol_second_stage = list(p.map(sub_problem_dual, model_second_stage))

            A_cuts = zeros((self.N, self.nx_first_stage + N_additional))
            b_cuts = zeros((self.N, 1))
            for i in range(self.N):
                # Solve the dual problem
                A_cuts[i, 0:self.nx_first_stage] = -transpose(
                    transpose(model_second_stage[i]["Ts"]).dot(sol_second_stage[i]["x"]))
                b_cuts[i, 0] = -transpose(sol_second_stage[i]["x"]).dot(model_second_stage[i]["hs"])

                if sol_second_stage[i]["status"] == 1:  # if the primal problem is feasible, add feasible cuts
                    A_cuts[i, self.nx_first_stage + i] = -1
                    obj_value_second_stage[i, 0] = sol_second_stage[i]["objvalue"]
                else:
                    obj_value_second_stage[i, 0] = inf
            # The var calculation is based on the following LP in problem (4) of ref[4]
            (VaRs,VAR) = BendersDecomposition.CVaR(self, ps, obj_value_second_stage,alpha)

            A_cuts_add = zeros((self.N, self.nx_first_stage + N_additional))
            b_cuts_add = zeros((self.N, 1))
            for i in range(self.N):
                # Solve the dual problem
                if sol_second_stage[i]["status"] == 1:  # if the primal problem is feasible, add feasible cuts
                    A_cuts_add[i, 0:self.nx_first_stage] = -transpose(
                    transpose(model_second_stage[i]["Ts"]).dot(sol_second_stage[i]["x"]))
                    b_cuts_add[i, 0] = -transpose(sol_second_stage[i]["x"]).dot(model_second_stage[i]["hs"])
                    A_cuts_add[i, self.nx_first_stage + self.N + i] = -1  # vi
                    A_cuts_add[i, self.nx_first_stage + self.N * 2] = -1  # CVAR
                    ## The CVAR should be estimiated here
            # This is a lazy way, in
            Upper = min(transpose(x_first_stage).dot(model_first_stage["c"]) + transpose(obj_value_second_stage).dot(ps) + \
                    ru*transpose(VaRs).dot(ps)+ ru*VAR,Upper)
            print("The upper bound is {0}".format(Upper[0][0]))

            Gap.append(BendersDecomposition.gap_calculaiton(self, Upper[-1], Lower))
            print("The gap is {0}".format(Gap[-1][0]))
            print("The lower bound is {0}".format(Lower))
            iter += 1

            if Gap[-1][0] < eps:
                break

        with Pool(n_processors) as p:
            sol_second_stage = list(p.map(sub_problem, model_second_stage))

        # x_first_stage = sol_first_stage["x"][0:self.nx_first_stage]
        #
        # x_second_stage = zeros((self.N, self.nx_second_stage))
        x_second_stage = [0] * self.N

        for i in range(self.N):
            x_second_stage[i] = array(sol_second_stage[i]["x"])

        sol = {"objvalue": Upper,
               "x_first_stage": x_first_stage,
               "x_second_stage": x_second_stage, }

        return sol

    def master_problem(self, model):
        """
        Solve the master problem
        :param model:
        :return:
        """
        (x, objvalue, status) = lp(model["c"], Aeq=model["Aeq"], beq=model["beq"], A=model["A"], b=model["b"],
                                   xmin=model["lb"], xmax=model["ub"], vtypes=model["vtypes"])

        sol = {"x": x,
               "objvalue": objvalue,
               "status": status}

        return sol

    def sub_problems_update(self, model, x):
        """

        :param model: The second stage models
        :param hs: The equality constraints under each stage
        :param Ts: The coupling constraints between the first stage and second stage constraints
        :return: hs-Ts*x
        """
        for i in range(self.N):
            model[i]["b"] = model[i]["hs"] - model[i]["Ts"].dot(x)

        return model

    def gap_calculaiton(self, upper, lower):

        if lower != 0:
            gap = abs((upper - lower) / lower * 100)
        else:
            gap = inf

        if gap == inf:
            gap = [inf]

        return gap
    def CVaR(self, ws, qs, alpha):
        """
        Define a supportive function for the calculation of CVaR
        :param ws:
        :param qs:
        :return:
        """
        nx = self.N
        lb = vstack([zeros((nx,1)),-ones((1,1))*self.M])
        c = vstack([ws/(1-alpha),ones((1,1))])
        A = zeros((self.N,nx+1))
        b = zeros((self.N,1))
        for s in range(self.N):
            A[s, s] = -1
            A[s, nx] = -1
            b[s, 0] = -qs[s]
        (sol,obj,status) = lp(c,A=A,b=b,xmin=lb)

        vari = sol[0:nx]
        VAR = sol[-1]
        return vari, VAR


def sub_problem(model):
    """
    Solve each slave problems
    :param model:
    :return:
    """
    (x, objvalue, status) = lp(model["c"], A=-model["A"], b=-model["b"])

    sol = {"x": x,
           "objvalue": objvalue,
           "status": status}

    return sol


def sub_problem_dual(model):
    """
    Solve each slave problems
    :param model:
    :return:
        """
    (x, objvalue, status) = lp_dual(model["b"], Aeq=transpose(model["A"]), beq=model["c"], xmin=zeros((len(model["b"]), 1)))

    sol = {"x": x,
           "objvalue": objvalue,
           "status": status}

    return sol


if __name__ == "__main__":
    # c = array([2, 3, 0, 0]).reshape(4, 1)
    # Ts = array([[1, 2, -1, 0], [2, -1, 0, -1]])
    # hs = array([3, 4]).reshape(2, 1)
    # Ws = array([1, 3]).reshape(2, 1)
    # lb = zeros((4, 1))
    # ub = ones((4, 1)) * inf
    # qs = array([2]).reshape(1, 1)
    # benders_decomposition = BendersDecomposition()
    # sol = benders_decomposition.main(c=c, lb=lb, ub=ub, ps=[1], qs=[qs], hs=[hs], Ts=[Ts], Ws=[Ws])
    # print(sol)

    # The second test case
    c = array([1, 1]).reshape(2, 1)
    lb = zeros((2, 1))

    ps = array([1 / 3, 1 / 3, 1 / 3]).reshape(3, 1)
    hs = [0] * 3
    hs[0] = array([7, 4]).reshape(2, 1)/2
    hs[1] = array([7, 4]).reshape(2, 1)/2
    hs[2] = array([7, 4]).reshape(2, 1)/2

    Ws = [0] * 3
    Ws[0] = array([[1, 0, -1, 0], [0, 1, 0, -1]])
    Ws[1] = array([[1, 0, -1, 0], [0, 1, 0, -1]])
    Ws[2] = array([[1, 0, -1, 0], [0, 1, 0, -1]])

    Ts = [0] * 3
    Ts[0] = array([[1, 1], [1 / 3, 1]])
    Ts[1] = array([[5 / 2, 1], [2 / 3, 1]])
    Ts[2] = array([[4, 1], [1, 1]])

    qs = [0] * 3
    qs[0] = array([1, 1, 0, 0])
    qs[1] = array([1, 1, 0, 0])
    qs[2] = array([1, 1, 0, 0])

    benders_decomposition = BendersDecomposition()
    sol = benders_decomposition.main(c=c, lb=lb, ps=ps, qs=qs, hs=hs, Ts=Ts, Ws=Ws, ru=0.5)
    print(sol)
