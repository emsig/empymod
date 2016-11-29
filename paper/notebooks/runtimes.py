
# coding: utf-8

# # Run time comparison
# 
# *This notebook is to reproduce the results from the article. For more information see the article in the parent directory.*

# In[1]:

import os
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from empymod.model import frequency

# Style adjustments
get_ipython().magic('matplotlib inline')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.style'] = 'normal'
colors = [color['color'] for color in list(mpl.rcParams['axes.prop_cycle'])]


# ## Estimate run times for QWE, FHT with 201 pt filter, and FHT with 801 pt filter
# 
# For the results as given in Table 2 in the manuscript.
# 
# `Matlab-file` to check run times for Key12:
# [data/Key12-Matlab-CompTimes.m](./data/Key12-Matlab-CompTimes.m)
# 
# The above file uses the following `timeit`-function:
# [data/timeit.m](./data/timeit.m)
# I used it with Matlab R2012b; as far as I understood, this `timeit` is backed into Matlab in newer versions.
# 
# Result with Matlab R2012b:
# [data/Key12-Matlab-CompTimes-Result.txt](./data/Key12-Matlab-CompTimes-Result.txt)

# In[2]:

# Loop over Hankel transforms
for HT in ['QWE', 'FHT1', 'FHT2']:
    
    if HT == 'FHT1':     # FHT 1: Key 201 pt from Key, 2012
        HT = 'FHT'
        print('** FHT 201')
        HTARG = 'key_201_2012'

    elif HT == 'FHT2':   # FHT 2: Anderson 801 pt from Key, 2012
        HT = 'FHT'
        print('** FHT 801')
        HTARG = 'anderson_801_1982'

    else:
        print('** QWE')
        
    # Loop over optimisation
    for opt in [None, 'spline']:
        if HT == 'QWE':
            if opt == 'spline':
                HTARG = [1e-2, 1e-24, 9, 40, 40]
            else:
                HTARG = [1e-6, 1e-24, 9]

        # Print which HT/optimisation used
        print('          ** opt = '+str(opt))

        # Loop over models
        for model in range(8):
            
            if model < 5:
                depth = np.array([0, 1000, 2000, 2100])                     # Layer top depths
                res = np.array([1e12, 0.3, 1, 100, 1])                      # Layer resistivities (ohm-m)
            else:
                depth = np.r_[0, 1000, 2000, 2100+np.linspace(0,10000,96)]  # Layer top depths
                res = np.r_[1e12, .3, 1, 100, np.ones(96)]                  # Layer resistivities (ohm-m)

            if model in [0,]:    # 1 offset
                noff = 1
            elif model in [1,]:  # 5 offsets
                noff = 5
            elif model in [2, 5]:  # 21 offsets
                noff = 21
            elif model in [3, 6]:  # 81 offsets
                noff = 81
            elif model in [4, 7]: # 321 offsets
                noff = 321
            rec   = [np.linspace(500, 20000, noff), np.zeros(noff), 1000]

            # Calculation
            out = get_ipython().magic('timeit -q -o frequency([0, 0, 990], rec, depth, res, freq=1, ab=11, xdirect=False, ht=HT, htarg=HTARG, opt=opt, verb=0)')
            print(u'%10.0f ms :: ' % (1000*out.best) + '  ** Layers ::', res.size, '; Offsets ::', noff, '**')

        print(' ')
    print(' ')


# ## Run time estimation of `Dipole1D` and `empymod` for 201 pt filter

# In[3]:

os.chdir('data')

# Loop over models
for model in range(8):
    if model < 5:
        depth = np.array([0, 1000, 2000, 2100])                     # Layer top depths
        res = np.array([1e12, 0.3, 1, 100, 1])                      # Layer resistivities (ohm-m)
    else:
        depth = np.r_[0, 1000, 2000, 2100+np.linspace(0,10000,96)]  # Layer top depths
        res = np.r_[1e12, .3, 1, 100, np.ones(96)]                  # Layer resistivities (ohm-m)

    if model in [0,]:    # 1 offset
        noff = 1
    elif model in [1,]:  # 5 offsets
        noff = 5
    elif model in [2, 5]:  # 21 offsets
        noff = 21
    elif model in [3, 6]:  # 81 offsets
        noff = 81
    elif model in [4, 7]: # 321 offsets
        noff = 321
    rec   = [np.linspace(500, 20000, noff), np.zeros(noff), 1000]
    src = [0, 0, 990]
    freq = 1

    # Print Model
    print(u'Layers ::', res.size, '; Offsets ::', noff)
    
    # Run empymod
    out0 = get_ipython().magic("timeit -q -o frequency(src, rec, depth, res, freq=freq, ab=11, xdirect=False, ht='FHT', htarg='key_201_2009', verb=0)")
    out1 = get_ipython().magic("timeit -q -o frequency(src, rec, depth, res, freq=freq, ab=11, xdirect=False, ht='FHT', opt='parallel', htarg='key_201_2009', verb=0)")
    out2 = get_ipython().magic("timeit -q -o frequency(src, rec, depth, res, freq=freq, ab=11, xdirect=False, ht='FHT', opt='spline', htarg='key_201_2009', verb=0)")

    print(u'                             empymod :: %5.0f ms' % (1000*out0.best))
    print(u'                    parallel empymod :: %5.0f ms' % (1000*out1.best))
    print(u'                      spline empymod :: %5.0f ms' % (1000*out2.best))

    # Write RUNFILE
    ffile = './RUNFILE'
    with open(ffile, 'wb') as runfile:
        runfile.write(bytes(
            'Version:          DIPOLE1D_1.0\n'
            'Output Filename:  dipole1d.csem\n'
            'HT Filters:       kk_ht_201\n'
            'UseSpline1D:      no\n'
            'CompDerivatives:  no\n'
            '# TRANSMITTERS:   1\n'
            '          X           Y           Z    ROTATION         DIP\n',
            'UTF-8'))
        np.savetxt(runfile, np.atleast_2d(np.r_[src[0], src[1], src[2], 0, 0]), fmt='%12.4f')
        runfile.write(bytes('# FREQUENCIES:    1\n', 'UTF-8'))
        np.savetxt(runfile, [freq,], fmt='%10.3f')
        runfile.write(bytes('# LAYERS:         '+str(np.size(depth)+1)+'\n', 'UTF-8'))
        np.savetxt(runfile, np.r_[[np.r_[-1000000, depth]], [res]].transpose(), fmt='%12.5g')
        runfile.write(bytes('# RECEIVERS:      '+str(np.size(rec[0]))+'\n', 'UTF-8'))
        rec = np.r_[[rec[0].ravel()], [rec[1].ravel()], [np.ones(np.size(rec[0]))*rec[2]]]
        np.savetxt(runfile, rec.transpose(), fmt='%12.4f')

    # Run DIPOLE1D
    out3 = get_ipython().magic("timeit -q -o subprocess.run('DIPOLE1D RUNFILE', shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)")
    print(u'                            DIPOLE1D :: %5.0f ms' % (1000*out3.best))

    # Write RUNFILE
    ffile = './RUNFILE'
    with open(ffile, 'wb') as runfile:
        runfile.write(bytes(
            'Version:          DIPOLE1D_1.0\n'
            'Output Filename:  dipole1d.csem\n'
            'HT Filters:       kk_ht_201\n'
            'UseSpline1D:      yes\n'
            'CompDerivatives:  no\n'
            '# TRANSMITTERS:   1\n'
            '          X           Y           Z    ROTATION         DIP\n',
            'UTF-8'))
        np.savetxt(runfile, np.atleast_2d(np.r_[src[0], src[1], src[2], 0, 0]), fmt='%12.4f')
        runfile.write(bytes('# FREQUENCIES:    1\n', 'UTF-8'))
        np.savetxt(runfile, [freq,], fmt='%10.3f')
        runfile.write(bytes('# LAYERS:         '+str(np.size(depth)+1)+'\n', 'UTF-8'))
        np.savetxt(runfile, np.r_[[np.r_[-1000000, depth]], [res]].transpose(), fmt='%12.5g')
        runfile.write(bytes('# RECEIVERS:      '+str(np.size(rec[0]))+'\n', 'UTF-8'))
        rec = np.r_[[rec[0].ravel()], [rec[1].ravel()], [np.ones(np.size(rec[0]))*rec[2]]]
        np.savetxt(runfile, rec.transpose(), fmt='%12.4f')

    # Run DIPOLE1D
    out4 = get_ipython().magic("timeit -q -o subprocess.run('DIPOLE1D RUNFILE', shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)")
    print(u'                     spline DIPOLE1D :: %5.0f ms' % (1000*out4.best))
    print('\n')
    
os.chdir('..')


# ### Check for subprocess
# 
# Run two instances, once with `timeit`, and once without, but printing the output from DIPOLE1D. Just to check if the times agree. As they are two different runs, they will differ slightly, but they should be close enough.

# In[4]:

os.chdir('data')

depth = np.r_[0, 1000, 2000, 2100+np.linspace(0,10000,96)]  # Layer top depths
res = np.r_[1e12, .3, 1, 100, np.ones(96)]                  # Layer resistivities (ohm-m)
noff = 321
rec   = [np.linspace(500, 20000, noff), np.zeros(noff), 1000]
src = [0, 0, 990]
freq = 1

# Print Model
print(u'Layers ::', res.size, '; Offsets ::', noff)

# Write RUNFILE
ffile = './RUNFILE'
with open(ffile, 'wb') as runfile:
    runfile.write(bytes(
        'Version:          DIPOLE1D_1.0\n'
        'Output Filename:  dipole1d.csem\n'
        'UseSpline1D:      no\n'
        'HT Filters:       kk_ht_201\n'
        'CompDerivatives:  no\n'
        '# TRANSMITTERS:   1\n'
        '          X           Y           Z    ROTATION         DIP\n',
        'UTF-8'))
    np.savetxt(runfile, np.atleast_2d(np.r_[src[0], src[1], src[2], 0, 0]), fmt='%12.4f')
    runfile.write(bytes('# FREQUENCIES:    1\n', 'UTF-8'))
    np.savetxt(runfile, [freq,], fmt='%10.3f')
    runfile.write(bytes('# LAYERS:         '+str(np.size(depth)+1)+'\n', 'UTF-8'))
    np.savetxt(runfile, np.r_[[np.r_[-1000000, depth]], [res]].transpose(), fmt='%12.5g')
    runfile.write(bytes('# RECEIVERS:      '+str(np.size(rec[0]))+'\n', 'UTF-8'))
    rec = np.r_[[rec[0].ravel()], [rec[1].ravel()], [np.ones(np.size(rec[0]))*rec[2]]]
    np.savetxt(runfile, rec.transpose(), fmt='%12.4f')

# Run DIPOLE1D
out = get_ipython().magic("timeit -q -o subprocess.run('DIPOLE1D RUNFILE', shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)")
print(u'Result from timeit/subprocess :: %10.0f ms' % (1000*out.best))

out2 = subprocess.run('DIPOLE1D RUNFILE', shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
#print(out2.stdout.decode("latin"))
print(u'Result from DIPOLE1D-output   :: ', out2.stdout.decode("latin")[-50:-1])

os.chdir('..')

