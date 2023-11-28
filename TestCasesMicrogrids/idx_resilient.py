"""
Hybrid AC/DC micro grids data format

Case format for two-stage stochastic optimization with multiple DGs, RESs and ESSs.
"""
NG = 2
NRES = 1
NESS = 1
PG0 = 0
FUEL0 = PG0 + NG
PUG = FUEL0 + NG
PBIC_A2D = PUG + 1
PBIC_D2A = PBIC_A2D + 1
PESS_CH0 = PBIC_D2A + 1
PESS_DC0 = PESS_CH0 + NESS
EESS0 = PESS_DC0 + 1
PPV0 = EESS0 + 1
PAC = PPV0 + 1  # The AC load sheding
PDC = PAC + 1  # The DC load sheding

NX_MG = PDC + 1
