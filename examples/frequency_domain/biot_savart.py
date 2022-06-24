# %% [markdown]
# Even though empymod was developed for CSEM modeling, the diverse inputs make it also possible to investigate line sources. The simplest way to investigate the magnetic field of line sources is by using Biot-Savart for an infinite wire. 
# 
# $$B(r,I) = \frac{μ_{0} I}{2\pi r}$$
# 
# Let us consider a line source approximated as a bipole with lots of source points, and compare it to the Biot Savart solution. 

# %% [markdown]
# Import packages 

# %%
import numpy as np
import matplotlib.pyplot as plt 
import empymod

# %% [markdown]
# Set model parameters. Currently, the line is in the xy plane at 0 depth. The line runs from 0 to 90 metres. This is arbitrary. However, do note that if you would like to investigate the behaviour of the bipole at further distances, the line/bipole should be extended with more source points for it to approximate an infinite wire. 

# %%
# Start and end coordinates line 
A = [0, 0, 0.0]           
B = [90, 0, 0.0]

I = 0.1                     # Current in Ampère. Scalar
freqtime = 83               # Frequency in Hz

# Modelling parameters
verb = 1
name = 'Biot savart vs. Empymod comparison'      # Model name

# Whole space depth res
whole_space_depth = []                 # Layer boundaries
whole_space_res = [3e8]                # Resistivities

# Half space depth res
half_space_depth = [0]                    # Layer boundaries
half_space_res = [3e8, 0.1]               # Resistivities

# Heights receivers
height = 1
# Number of receivers 
nrec = 41
rec_posi = [1, 20]  # receivers range from posi 1 to posi 2 
srcpts = 401

# To get from H to B: B = mu0 * H
mu_0 = 4*np.pi*10**-7

# indicates direction field measured
rec_y = [np.ones(nrec)*(B[0] - A[0])/2, np.linspace(rec_posi[0], rec_posi[1], nrec), np.ones(nrec), 90, 0]
rec_z = [np.ones(nrec)*(B[0] - A[0])/2, np.linspace(rec_posi[0], rec_posi[1], nrec), np.ones(nrec), 0, -90]

angles = np.arctan(height/rec_y[1])

R = np.sqrt(height**2 + rec_y[1]**2)

# Different input data for different wires regarding the model: ws = whole space, hs = half space
inpdat_ws_AB = {'src': [A[0], B[0], A[1], B[1], A[2], B[2]], 'depth': whole_space_depth, 'res':whole_space_res, 'freqtime': freqtime,
           'srcpts': srcpts, 'mrec':True, 'strength': I}

inpdat_hs_AB = {'src': [A[0], B[0], A[1], B[1], A[2], B[2]], 'depth': half_space_depth, 'res':half_space_res, 'freqtime': freqtime,
           'srcpts': srcpts, 'mrec':True, 'strength': I}

# %% [markdown]
# Now we can compute the solutions for the "infinite" wires. Note that to acquire the different components, one should adjust the receiver orientation. The appropriate combinations are: (90, 0) = y-dir,   (0, 0) = x-dir,   ('theta', -90) = z-dir.  (always something, -90 is Z because azimuth then doesn't matter, minus sign due to RHS system). 

# %%
# Compute for different directions 
wire_wsy = empymod.bipole(**inpdat_ws_AB, rec = rec_y)*mu_0
wire_wsz = empymod.bipole(**inpdat_ws_AB, rec = rec_z)*mu_0

wire_hsy = empymod.bipole(**inpdat_hs_AB, rec = rec_y)*mu_0
wire_hsz = empymod.bipole(**inpdat_hs_AB, rec = rec_z)*mu_0

# %% [markdown]
# To investigate the angle between the Biot savart and empymod bipole solution, one needs the total field to produce the unit vectors of both solutions.

# %%
# Calculate total magnetic fields
wire_tot_ws = np.sqrt(wire_wsy**2 + wire_wsz**2)
wire_tot_hs = np.sqrt(wire_hsy**2 + wire_hsz**2)

# %%
# Unit vectors 
unitvec_wire_ws = [abs(wire_wsy), abs(wire_wsz)]/abs(wire_tot_ws)
unitvec_wire_hs = [abs(wire_hsy), abs(wire_hsz)]/abs(wire_tot_hs) 

# %%
# Biot Savart 
Biot_savart_tot = mu_0*I/(2*np.pi*R)
Biot_savart_y = np.sin(angles)*Biot_savart_tot
Biot_savart_z = np.cos(angles)*Biot_savart_tot

# Unit vectors BS
uv_biot_savart = [Biot_savart_y, Biot_savart_z]/Biot_savart_tot


# %%
# Angle difference between BS and empymod 
angle_diff_bs_ws = np.zeros(nrec)
angle_diff_bs_hs = np.zeros(nrec)
for j in range(nrec):
    angle_diff_bs_ws[j] = np.arccos(np.dot(uv_biot_savart[:,j], unitvec_wire_ws[:,j]))*180/np.pi
    angle_diff_bs_hs[j] = np.arccos(np.dot(uv_biot_savart[:,j], unitvec_wire_hs[:,j]))*180/np.pi

angle_diff_bs_ws[np.isnan(angle_diff_bs_ws)] = 0

# %% [markdown]
# Plotting the Biot Savart and empymod.bipole full and half space solutions. What is noticeable is that the Biot Savart and full space solution, as well as the Z field component from all three solutions, are close to identical. The biggest change is related to the Y component in the half space solution. This is related to the relative TE and TM modes. The Z field is determined by the TE mode and the Y to TM. Recalling the reflection coefficients: 
# 
# $$r_{n}^{T M} = \frac{\eta_{n+1} \Gamma_{n}-\eta_{n} \Gamma_{n+1}}{\eta_{n+1} \Gamma_{n}+\eta_{n} \Gamma_{n+1}}$$
# 
# $$r_{n}^{T E} = \frac{\zeta_{n+1} \Gamma_{n}-\zeta_{n} \Gamma_{n+1}}{\zeta_{n+1} \Gamma_{n}+\zeta_{n} \Gamma_{n+1}}$$
# 
# 
# Since the conductivity of air almost zero, in a half-space the local reflection coefficient $r^{TM} = 1$. At the same time, $r^{TE}$ is dominated by $μ$ which is similar in air as in any soil, making the reflection coefficient for the TE mode equal to zero. 

# %%
# Plot total fields. 
fig, axs = plt.subplots(2, 2, figsize=(22, 14))
axs1, axs2, axs3, axs4 = axs[0, 0], axs[1, 0], axs[0, 1],axs[1, 1]

axs1.set_title('Modelled Z-field away from the cable', fontsize = 20)
axs1.set_xlabel('Receiver distance (m)', fontsize = 16)
axs1.set_ylabel('Amplitude (nT)', fontsize = 16)
axs1.plot(R, abs(wire_wsz) * 10**9, 'C3+-', markersize= 10, label='Wire ws')
axs1.plot(R, abs(wire_hsz) * 10**9, 'C5x-', markersize= 9, label='Wire hs')
axs1.plot(R, abs(Biot_savart_z) *10**9, 'C1o-', markersize= 3, label='Biot Sav')
axs1.set_xlim([0, 15])
axs1.set_ylim([0, 15])
axs1.legend(loc='best', fontsize = 14)

axs2.set_title('Y-field', fontsize = 20)
axs2.set_xlabel('Receiver distance (m)', fontsize = 16)
axs2.set_ylabel('Amplitude (nT)', fontsize = 16)
axs2.plot(R, abs(wire_wsy) *10**9, 'C3+-',  markersize= 10, label='Wire ws')
axs2.plot(R, abs(wire_hsy) *10**9, 'C5x-', markersize= 9, label='Wire hs')
axs2.plot(R, abs(Biot_savart_y) *10**9, 'C1o-', markersize= 3, label='Biot Sav')
axs2.set_xlim([0, 15])
axs2.set_ylim([0, 15])
axs2.legend(loc='best', fontsize = 14)

axs3.set_title('Total magnetic field away from the cable', fontsize = 20)
axs3.set_xlabel('Receiver distance (m)', fontsize = 16)
axs3.set_ylabel('Amplitude (nT)', fontsize = 16)
axs3.plot(R, abs(wire_tot_ws) *10**9, 'C3+-',  markersize= 10, label='Wire ws')
axs3.plot(R, abs(wire_tot_hs) *10**9, 'C5x-',  markersize= 9, label='Wire hs')
axs3.plot(R, abs(Biot_savart_tot) *10**9, 'C1o-', markersize= 3, label='Biot Sav')
axs3.set_xlim([0, 15])
axs3.set_ylim([0, 15])
axs3.legend(loc='best', fontsize = 14)

axs4.set_title('Angle difference Biot Savart and Half space solution', fontsize = 20)
axs4.set_xlabel('Receiver distance (m)', fontsize = 16)
axs4.set_ylabel('Angle (degrees)', fontsize=16)
axs4.plot(R, abs(angle_diff_bs_hs), 'C3+-', markersize= 8)
plt.show()

fig.savefig('Biot_savart_bipole_comp.svg', format='svg')


# %% [markdown]
# The total field and Y and Z components are compared. The bottom right shows the angle difference between the vectors of the Biot-Savart and bipole half-space solution.
# 
# If we take a finite length wire, eventually Biot-Savart for an infinite length wire will not apply. The question is: at what receiver distance relative to the finite wire is Biot Savart still decently applicable. When testing this in empimod and comparing Biot-Savart for an infinite line source to a finite length wire, it is has been concluded that when the receiver is a 1/10th of the length of the wire away from its middle, Biot Savart and empimod agree within 2\%. For 1\% accuracy between the two methods, the receiver distance should not extend more than 7\% of the length of the wire. 


