r"""Create data for test_fdesign."""
import numpy as np
from copy import deepcopy as dc
from empymod.scripts import fdesign

# Cannot pickle/shelve this; could dill it. For the moment, we just provide
# it separately here and in the tests.
fI = (fdesign.j0_1(5), fdesign.j1_1(5))

# Define main model
inp1 = {'spacing': (0.04, 0.1, 10),
        'shift': (-3, -0.5, 10),
        'n': 201,
        'cvar': 'amp',
        'save': False,
        'full_output': True,
        'r': np.logspace(0, 3, 10),
        'r_def': (1, 1, 2),
        'name': 'test',
        'finish': None,
        }

# 1. General case with various spacing and shifts
filt1, out1 = fdesign.design(verb=0, plot=0, fI=fI, **inp1)
case1 = (inp1, filt1, out1)

# 2. Specific model with only one spacing/shift
inp2 = dc(inp1)
inp2['spacing'] = 0.0641
inp2['shift'] = -1.2847
filt2, out2 = fdesign.design(verb=0, plot=0, fI=fI, **inp2)
case2 = (inp2, filt2, out2)

# 3 Same, with only one transform
filt3, out3 = fdesign.design(verb=0, plot=0, fI=fI[0], **inp2)
case3 = (inp2, filt3, out3)

# 4. Maximize r
inp4 = dc(inp2)
inp4['cvar'] = 'r'
filt4, out4 = fdesign.design(verb=0, plot=0, fI=fI, **inp4)
case4 = (inp4, filt4, out4)

# 5. One shift, several spacings
inp5 = dc(inp1)
inp5['spacing'] = (0.06, 0.07, 10)
inp5['shift'] = -1.2847
filt5, out5 = fdesign.design(verb=0, plot=0, fI=fI, **inp5)
case5 = (inp5, filt5, out5)

# 6. Several shifts, one spacings; r
inp6 = dc(inp1)
inp6['spacing'] = 0.0641
inp6['shift'] = (-1.5, 0, 10)
inp6['cvar'] = 'r'
filt6, out6 = fdesign.design(verb=0, plot=0, fI=fI, **inp6)
case6 = (inp6, filt6, out6)

# # Store data # #
np.savez_compressed('../data/fdesign.npz',
                    case1=case1, case2=case2, case3=case3,
                    case4=case4, case5=case5, case6=case6)
