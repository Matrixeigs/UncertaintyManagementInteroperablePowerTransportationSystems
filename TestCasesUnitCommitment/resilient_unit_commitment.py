"""
Resilient unit commitment using the matrix formulated by Matlab
It
"""

import scipy.io
from RobustOptimization.two_stage_robust_optimization_version2 import TwoStageRobustOptimization
# from StochasticOptimization.ccg_benders_decomposition import BendersDecomposition
from StochasticOptimization.benders_decomposition import BendersDecomposition
# from StochasticOptimization.benders_decomposition_cvar import BendersDecomposition
import numpy as np

class ResilientUnitCommitment():
    def run(self):
        ## Test case 1: for the robust optimization part
        # data = scipy.io.loadmat('UC_RO_N_k.mat')
        # ## Read data from matlab file
        # cx = data['cx']
        # Aineq = data['Aineq'].toarray()
        # bineq = data['bineq']
        # Aeq = data['Aeq'].toarray()
        # beq = data['beq']
        # lx = data['lx']
        # ux = data['ux']
        # vtypes = data['vtypes'].tolist()
        #
        # cy = data['cy']
        # G = data['G']
        # E = data['E']
        # M = data['M']
        # h = data['h']
        # Cu = data['Cu']
        # fu = data['fu']
        # ## The robust optimization solution
        # two_stage_robust_optimization = TwoStageRobustOptimization()
        # result = two_stage_robust_optimization.main(cx,Aeq=Aeq,beq=beq,A=Aineq,b=bineq,lb=lx,ub=ux,vtypes=vtypes,d=cy,G=E,E=G,M=M,h=h,Cu=Cu,fu=fu)


        ## Test case 2: for the stochastic optimization part
        data = scipy.io.loadmat('UC_SO_N_k.mat')
        ## Read data from matlab file
        cx = data['cx']
        Aineq = data['Aineq'].toarray()
        bineq = data['bineq']
        Aeq = data['Aeq'].toarray()
        beq = data['beq']
        lx = data['lx']
        ux = data['ux']
        vtypes = data['vtypes'].tolist()

        cy = data['cy']
        G = data['G']
        E = data['E']
        M = data['M']
        h = data['h']
        # Cu = data['Cu']
        # fu = data['fu']
        ## For scenario
        u_sample = data['u_sample']
        Ns = data['Ns']
        ws = data['ws']
        hs = []
        Ws = []
        Ts = []
        qs = []
        for i in range(int(Ns)):
            hs.append(h - np.reshape(np.dot(M, u_sample[:, i]), (-1, 1)))
            qs.append(cy)
            Ts.append(G)
            Ws.append(E)
        # The stochastic optimization solution
        benders_decomposition = BendersDecomposition()
        result_so = benders_decomposition.main(cx,A=Aineq,b=bineq,Aeq=Aeq,beq=beq,lb=lx,ub=ux,vtype=vtypes,ps=ws,qs=qs,Ts=Ts,Ws=Ws,hs=hs)

        ## Test case 3: for the risk-aversion stochastic optimization part
        ## Introducing additional parameters, the confidential level and weight factor
        # alpha = 0.9 # The confidential level
        # ro = 0.5  # weight factor on conditional value at risk
        # benders_decomposition = BendersDecomposition()
        # result_so = benders_decomposition.main(cx, A=Aineq, b=bineq, Aeq=Aeq, beq=beq, lb=lx, ub=ux, vtype=vtypes,
        #                                        ps=ws, qs=qs, Ts=Ts, Ws=Ws, hs=hs,alpha=alpha,ru=ro)

        ##
        ## Result analysis function


if __name__=="__main__":
    resilient_unit_commitment=ResilientUnitCommitment()
    resilient_unit_commitment.run()