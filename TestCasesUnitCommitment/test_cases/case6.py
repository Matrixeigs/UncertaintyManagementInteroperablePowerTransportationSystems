"""
Units data for wind hydro dispatch
Data is obtained from real test systems

"""
from numpy import array


def case6():
    """Power flow data for real wind hydro power systems

    @return: Power flow data for jointed wind hydro power systems
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 20, 23, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [2, 2, 40, 38.5, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [3, 2, 40, 38.5, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [4, 2, 0, 0, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [5, 1, 0, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 0, 0, 200, -80, 1, 100, 1, 220, 100, 0, 0, 0, 0, 0, 0, 55 / 12, 55 / 6, 55, 0, 0],
        [2, 100, 0, 70, -40, 1, 100, 1, 100, 10, 0, 0, 0, 0, 0, 0, 50 / 12, 50 / 6, 50, 0, 0],
        [6, 100, 0, 50, -40, 1, 100, 1, 20, 10, 0, 0, 0, 0, 0, 0, 20 / 12, 20 / 6, 20, 0, 0],
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 2, 0.0050, 0.170, 0.00, 200, 300, 300, 0, 0, 1, -360, 360],
        [2, 3, 0.0000, 0.037, 0.00, 110, 300, 300, 0, 0, 1, -360, 360],
        [1, 4, 0.0030, 0.258, 0.00, 100, 300, 300, 0, 0, 1, -360, 360],
        [2, 4, 0.0070, 0.197, 0.00, 100, 300, 300, 0.69, 0, 1, -360, 360],
        [4, 5, 0.0000, 0.037, 0.00, 100, 300, 300, 0.69, 0, 1, -360, 360],
        [5, 6, 0.0020, 0.140, 0.00, 100, 130, 130, 0, 0, 1, -360, 360],
        [3, 6, 0.0000, 0.018, 0.00, 100, 130, 130, 0, 0, 0, -360, 360],
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 10, 0, 3, 0.0004, 13.51476154, 176.9507820, 4, 4, 4],
        [2, 200, 0, 3, 0.001, 32.63061346, 129.9709568, 2, 3, 3],
        [2, 100, 0, 3, 0.005, 17.69711347, 137.4120219, 1, 1, 0],
    ])

    return ppc
