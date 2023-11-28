#=
Benders decomposition method for two-stage stochastic optimization problems
    Minimize a function F(X) beginning, subject to
	optional linear and nonlinear constraints and variable bounds:
			min  c'*x + sum(p_s*Q_s(x))
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
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 28 Nov 2023
@version: 1.0

notes:
1) The data structure is based on the numpy and scipy
2) This algorithm should be extended for further version to solve the jointed chance constrained stochastic programming
3) In this test algorithm, Mosek is adpoted. https://www.mosek.com/
4) In the second stage optimization, the dual problem is solved, so that only the cplex problem is needed to solve the problem.
5) The multi-cuts version Benders decomposition is adopted.
6) Dual cuts methods are adopted

=#
