# Copyright (c) 2017-2018. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
Power flow data for 123 bus test system under three phase balance scenario.
"""

from numpy import array
import pandas as pd

def case69():
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 10.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = pd.read_csv('case69bus.csv', header=None).values
    ppc["branch"] = pd.read_csv('case69branch.csv', header=None).values
    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf, start-up time, shut-down time and initial condition!
    ppc["gen"] = pd.read_csv('case69gen.csv', header=None).values

    ## branch data

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 8],
        [2, 23],
        [3, 26],
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = pd.read_csv('case69gencost.csv', header=None).values
    return ppc
