import emg3d
import numpy as np
from empymod.ckernel import fields
from numpy.testing import assert_allclose
data = emg3d.load('data.h5')['data']

## Cases

# Depths: `[-inf, 0, 150, 300, 500, 600]`
#
# | case | lrec | lsrc | zsrc |
# |------|:----:|:----:|:----:|
# |    1 |  5   |  1   |  100 |
# |    2 |  3   |  3   |  350 |
# |    3 |  0   |  5   |  700 |
# |    4 |  5   |  5   |  700 |
# |    5 |  5   |  0   |  -30 |

tP, sPud, sPu, sPd = 0, 0, 0, 0

# === fail ===
# - True, will stop at the first non-equal element and print the difference.
# - False, will not stop and print the result for all tests
# fail = True
fail = False

print(f" ab  TM/TE  case  ::  samePu  samePd\n{38*'-'}")
for ii, (_, val) in enumerate(data.items()):
    for i in range(1, 6):
        ab = int(val['ab'])
        TM = bool(val['TM'])
        Pu, Pd = fields(ab=ab, TM=TM, **val[f'inp-{i}'])
        samePu = np.allclose(Pu, val[f'out-{i}'][0, ...], rtol=1e-7, atol=0)
        samePd = np.allclose(Pd, val[f'out-{i}'][1, ...], rtol=1e-7, atol=0)
        tP += 1
        if samePu and samePd:
            sPud += 1
        elif samePu:
            sPu += 1
        elif samePd:
            sPd += 1

        if fail:
            if not samePu*samePd:
                print(f"ab: {ab}; TM: {TM}  ::    Pu: {samePu}    Pd: {samePd}\n{70*'-'}")
                assert_allclose(Pu, val[f'out-{i}'][0, ...])
                assert_allclose(Pd, val[f'out-{i}'][1, ...])
        else:
            print(f" {ab}    {['TE', 'TM'][TM]}     {i}   ::   {samePu!s:>5}   {samePd!s:>5}")


print(f"{70*'-'}\n  Total tests: {tP}x2; completely passed: {sPud}, only Pu: {sPu}, only Pd: {sPd}")



# Data Creation

# DATAKERNEL = np.load('/home/dtr/Codes/empymod/tests/data/kernel.npz', allow_pickle=True)
#
# data = DATAKERNEL['fields'][()]
#
# ndat = {}
# for k, v in data.items():
#     ndat[str(k)] = {
#         'ab': v[0],
#         'TM': v[1],
#         'inp-1': v[2],
#         'out-1': v[3],
#         'inp-2': v[4],
#         'out-2': v[5],
#         'inp-3': v[6],
#         'out-3': v[7],
#         'inp-4': v[8],
#         'out-4': v[9],
#         'inp-5': v[10],
#         'out-5': v[11],
#     }
#
# emg3d.save('/home/dtr/Desktop/data.h5', data=ndat)
