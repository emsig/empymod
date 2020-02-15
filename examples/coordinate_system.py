r"""
Coordinate system
=================

The derivation on which ``empymod`` is based ([HuTS15]_) and the actual
implementation consider a coordinate system where positive :math:`z` points
into the ground. This defines a left-handed system assuming that Easting is the
:math:`x`-direction and Northing is the :math:`y`-direction. However,
``empymod`` can equally be used for a coordinate system where positive
:math:`z` is pointing up by just flipping :math:`z`, keep Easting and Northing
the same.

+----------------+--------------------+---------------------+
|                | Left-handed system | Right-handed system |
+================+====================+=====================+
| :math:`x`      | Easting            | Easting             |
+----------------+--------------------+---------------------+
| :math:`y`      | Northing           | Northing            |
+----------------+--------------------+---------------------+
| :math:`z`      | Down               | Up                  |
+----------------+--------------------+---------------------+
| :math:`\theta` | Angle E-N          | Angle E-N           |
+----------------+--------------------+---------------------+
| :math:`\varphi`| Angle down         | Angle up            |
+----------------+--------------------+---------------------+

There are a few other important points to keep in mind when switching between
coordinate systems:

- The interfaces (``depth``) have always to be defined from lowest to highest.
  E.g., a simple three-layer model with the sea-surface at 0 m and 100 m water
  column is either defined as ``depth=[0, 100]`` for the LHS, or as
  ``depth=[-100, 0]`` for the RHS.
- The above statement affects also all model parameters (``resistivity``,
  ``anisotropy``, etc.). E.g., for the above three-layer example,
  ``res=[1e12, 0.3, 1]`` (air, water, subsurface) for the LHS, or ``res=[1,
  0.3, 1e12]`` (subsurface, water, air) for the RHS.
- A source or a receiver *exactly on* a boundary is taken as being in the lower
  layer. Hence, if :math:`z_\rm{rec} = z_0`, where :math:`z_0` is the surface,
  then the receiver is taken as in the air in the LHS, but as in the subsurface
  in the RHS. Similarly, if :math:`z_\rm{rec} = z_\rm{seafloor}`, then the
  receiver is taken as in the sea in the LHS, but as in the subsurface in the
  RHS. This can be avoided by never placing it exactly on a boundary, but
  slightly (e.g., 1 mm) in the layer where you want to have it.
- With ``dipole``, the ``ab``'s containing vertical directions (``3``) switch
  the sign for each vertical component.


In this example we first create a sketch of the LHS and RHS for visualization,
followed by a few examples using ``dipole`` and ``bipole`` to demonstrate the
two possibilities.
"""
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

###############################################################################
# RHS vs LHS
# ----------
#
# Comparison of the right-handed system with positive :math:`z` downwards and
# the left-handed system with positive :math:`z` upwards. Easting is always
# :math:`x`, and Northing is :math:`y`.

x = np.arange(11)
ang = np.deg2rad(45)
azimuth = np.deg2rad(70)
azarc = np.linspace(np.deg2rad(90), azimuth, 11)
dparc = np.linspace(np.deg2rad(90), np.deg2rad(125), 11)
dip = np.deg2rad(35)

xscale = 1.3
yscale = 3
zscale = 2

x0 = x
x1 = np.zeros(x.size)

y0 = np.sin(ang)*x
y1 = np.cos(ang)*x/yscale

z0 = np.zeros(x.size)
z1 = x/zscale

az0 = np.sin(azimuth)*x/xscale
az1 = np.cos(azimuth)*x/yscale/xscale

dp0 = np.cos(dip)*x/xscale
dp1 = np.sin(-dip)*x/xscale

fontdic = {
    'fontsize': 14,
    'horizontalalignment': 'center',
    'verticalalignment': 'center',
}
fontdic2 = fontdic.copy()
fontdic2['fontsize'] = 20

# FIGURE
plt.figure(figsize=(10, 5))

# # # Left-Handed System LHS # # #
ax1 = plt.subplot(121)
plt.title('Left-handed system (LHS)\nfor positive $z$ downwards', fontsize=14)

# x
plt.plot(x0, x1, 'k')
plt.text(x0[-1], x1[-1]-0.5, 'x', **fontdic)

# y
plt.plot(y0, y1, 'k')
plt.text(y0[-1]-0.5, y1[-1]/yscale+1.8, 'y', **fontdic)

# z
plt.plot(z0, -z1, 'k')
plt.text(-0.5, -z1[-1], 'z', **fontdic)

# azimuth
plt.plot(az0, az1, 'C1')
plt.plot(np.sin(azarc)*6, np.cos(azarc)*6/yscale, 'C1--')
plt.text(7.5, 0.5, r'$\theta$', color='C1', **fontdic2)

# dip
plt.plot(dp0, dp1, 'C0')
plt.plot(np.sin(dparc)*5, np.cos(dparc)*5, 'C0--')
plt.text(5.5, -1.5, r'$\varphi$', color='C0', **fontdic2)
plt.plot()

plt.axis('equal')
ax1.axis('off')

# # # right-Handed System RHS # # #
ax2 = plt.subplot(122, sharey=ax1, frameon=False)
plt.title('Right-handed system (RHS)\nfor positive $z$ upwards', fontsize=14)

# x
plt.plot(x0, x1, 'k')
plt.text(x0[-1], x1[-1]-0.5, 'x', **fontdic)

# y
plt.plot(y0, y1, 'k')
plt.text(y0[-1]-0.5, y1[-1]/yscale+1.8, 'y', **fontdic)

# z
plt.plot(z0, z1, 'k')
plt.text(-0.5, z1[-1], 'z', **fontdic)

# azimuth
plt.plot(az0, az1, 'C1')
plt.plot(np.sin(azarc)*6, np.cos(azarc)*6/yscale, 'C1--')
plt.text(7.5, 0.5, r'$\theta$', color='C1', **fontdic2)

# dip
plt.plot(dp0, dp1, 'C0')
plt.plot(np.sin(dparc)*5, np.cos(dparc)*5, 'C0--')
plt.text(5.5, -1.5, r'$-\varphi$', color='C0', **fontdic2)
plt.plot()

plt.axis('equal')
ax2.axis('off')

plt.show()


###############################################################################
# Dipole
# ------
#
# A simple example using ``dipole``. It is a marine case with 300 meter water
# depth and a 50 m thick target 700 m below the seafloor.
#
off = np.linspace(500, 10000, 301)


###############################################################################
# LHS
# ```
#
# In the left-handed system positive :math:`z` is downwards. So we have to
# define our model by beginning with the air layer, followed by water,
# background, target, and background again. This means that all our
# depth-values are positive, as the air-interface :math:`z_0` is at 0 m.

lhs = empymod.dipole(
        src=[0, 0, 100],
        rec=[off, np.zeros(off.size), 200],
        depth=[0, 300, 1000, 1050],
        res=[1e20, 0.3, 1, 50, 1],
        freqtime=1,
        verb=0
)


###############################################################################
# RHS
# ```
#
# In the right-handed system positive :math:`z` is upwards. So we have to
# define our model by beginning with the background, followed by the target,
# background again, water, and air. This means that all our depth-values are
# negative.

rhs = empymod.dipole(
        src=[0, 0, -100],
        rec=[off, np.zeros(off.size), -200],
        depth=[-1050, -1000, -300, 0],
        res=[1, 50, 1, 0.3, 1e20],
        freqtime=1,
        verb=0
)


###############################################################################
# Compare
# ```````
#
# Plotting the two confirms that the results agree, no matter if we use the LHS
# or the RHS definition.

plt.figure(figsize=(9, 4))

ax1 = plt.subplot(121)
plt.title('Real')
plt.plot(off/1e3, lhs.real, 'C0', label='LHS +')
plt.plot(off/1e3, -lhs.real, 'C4', label='LHS -')
plt.plot(off/1e3, rhs.real, 'C1--', label='RHS +')
plt.plot(off/1e3, -rhs.real, 'C2--', label='RHS -')
plt.yscale('log')
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.legend()

ax2 = plt.subplot(122, sharey=ax1)
plt.title('Imaginary')
plt.plot(off/1e3, lhs.imag, 'C0', label='LHS +')
plt.plot(off/1e3, -lhs.imag, 'C4', label='LHS -')
plt.plot(off/1e3, rhs.imag, 'C1-.', label='RHS +')
plt.plot(off/1e3, -rhs.imag, 'C2:', label='RHS -')
plt.yscale('log')
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.legend()
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

plt.tight_layout()
plt.show()


###############################################################################
# Bipole [x1, x2, y1, y2, z1, z2]
# -------------------------------
#
# A time-domain example using rotated bipoles, where we define them as
# :math:`[x_1, x_2, y_1, y_2, z_1, z_2]`.

times = np.linspace(0.1, 10, 301)
inp = {'freqtime': times, 'signal': 0, 'verb': 0}

lhs = empymod.bipole(
        src=[-50, 50, -10, 10, 100, 110],
        rec=[6000, 6100, 20, -20, 220, 200],
        depth=[0, 300, 1000, 1050],
        res=[1e20, 0.3, 1, 50, 1],
        **inp
)

rhs = empymod.bipole(
        src=[-50, 50, -10, 10, -100, -110],
        rec=[6000, 6100, 20, -20, -220, -200],
        depth=[-1050, -1000, -300, 0],
        res=[1, 50, 1, 0.3, 1e20],
        **inp
)

plt.figure()

plt.plot(times, lhs, 'C0', label='LHS')
plt.plot(times, rhs, 'C1--', label='RHS')
plt.xlabel('Time (s)')
plt.ylabel('$E_x$ (V/m)')
plt.legend()

plt.show()


###############################################################################
# Bipole [x, y, z, azimuth, dip]
# ------------------------------
#
# A very similar time-domain example using rotated bipoles, but this time
# defining them as :math:`[x, y, z, \theta, \varphi]`. Note that
# :math:`\varphi` has to change the sign, while :math:`\theta` does not.

lhs = empymod.bipole(
        src=[0, 0, 100, 10, 20],
        rec=[6000, 0, 200, -5, 15],
        depth=[0, 300, 1000, 1050],
        res=[1e20, 0.3, 1, 50, 1],
        **inp
)

rhs = empymod.bipole(
        src=[0, 0, -100, 10, -20],
        rec=[6000, 0, -200, -5, -15],
        depth=[-1050, -1000, -300, 0],
        res=[1, 50, 1, 0.3, 1e20],
        **inp
)

plt.figure()

plt.plot(times, lhs, 'C0', label='LHS')
plt.plot(times, rhs, 'C1--', label='RHS')
plt.xlabel('Time (s)')
plt.ylabel('$E_x$ (V/m)')
plt.legend()

plt.show()


###############################################################################

empymod.Report()
