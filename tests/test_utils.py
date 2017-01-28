# utils. Status: 14/21
import pytest
import numpy as np
from numpy.testing import assert_allclose

from empymod import utils


def test_emarray():                                                # 1. EMArray
    out = utils.EMArray(3)
    assert out.amp == 3
    assert out.pha == 0
    assert out.real == 3
    assert out.imag == 0

    out = utils.EMArray(1, 1)
    assert out.amp == np.sqrt(2)
    assert out.pha == 45.
    assert out.real == 1
    assert out.imag == 1

    out = utils.EMArray([1, 0], [1, 1])
    assert_allclose(out.amp, [np.sqrt(2), 1])
    assert_allclose(out.pha, [45., 90.])
    assert_allclose(out.real, [1, 0])
    assert_allclose(out.imag, [1, 1])


def test_check_ab(capsys):                                        # 2. check_ab
    # This is another way how check_ab could have been done: hard-coded.
    # We use it here to check the output of check_ab.
    iab = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26,
           31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46,
           51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66]
    oab = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26,
           31, 32, 33, 34, 35, 36, 14, 24, 34, 11, 12, 13,
           15, 25, 35, 21, 22, 23, 16, 26, 36, 31, 32, 33]
    omsrc = np.array([[False, ]*3 + [True, ]*3]*6).ravel()
    omrec = [False, ]*18 + [True, ]*18
    for i in range(len(iab)):
        ab, msrc, mrec = utils.check_ab(iab[i], 0)
        assert ab == oab[i]
        assert msrc == omsrc[i]
        assert mrec == omrec[i]

    utils.check_ab(36, 3)
    out, _ = capsys.readouterr()
    outstr = "   Input ab        :  36\n\n>  <ab> IS 36 WHICH IS ZERO; "
    assert out == outstr + "returning\n"

    utils.check_ab(44, 3)
    out, _ = capsys.readouterr()
    outstr = "   Input ab        :  44\n   Calculated ab   :  11\n"
    assert out == outstr

    # Check it raises a ValueError if a non-existing ab is provided.
    with pytest.raises(ValueError):
        utils.check_ab(77, 0)

    # We just check one other thing here, that it fails with a TypeError if a
    # list instead of one value is provided. Generally the try/except statement
    # with int() should take proper care of all the checking right in check_ab.
    with pytest.raises(TypeError):
        utils.check_ab([12, ], 0)


# 3. check_bipole


def test_check_dipole(capsys):                                # 4. check_dipole
    # correct input, verb > 2, src
    src, nsrc = utils.check_dipole([[1000, 2000], [0, 0], 0], 'src', 3)
    out, _ = capsys.readouterr()
    assert nsrc == 2
    assert_allclose(src[0], [1000, 2000])
    assert_allclose(src[1], [0, 0])
    assert_allclose(src[2], 0)

    outstr = "   Source(s)       :  2 dipole(s)\n"
    outstr += "     > x       [m] :  1000.0 - 2000.0 : 2  [min-max; #]\n"
    outstr += "     > y       [m] :  0.0 - 0.0 : 2  [min-max; #]\n"
    outstr += "     > z       [m] :  0.0\n"
    assert out == outstr

    # correct input, verb > 2, rec
    rec, nrec = utils.check_dipole([0, 0, 0], 'rec', 3)
    out, _ = capsys.readouterr()
    assert nrec == 1
    assert_allclose(rec[0], 0)
    assert_allclose(rec[1], 0)
    assert_allclose(rec[2], 0)

    outstr = "   Receiver(s)     :  1 dipole(s)\n"
    outstr += "     > x       [m] :  0.0\n"
    outstr += "     > y       [m] :  0.0\n"
    outstr += "     > z       [m] :  0.0\n"
    assert out == outstr

    # Check Errors: more than one z
    with pytest.raises(ValueError):
        utils.check_dipole([[0, 0], [0, 0], [0, 0]], 'src', 3)
    # Check Errors: wrong number of elements
    with pytest.raises(ValueError):
        utils.check_dipole([0, 0, 0, 0], 'rec', 3)


def test_check_frequency(capsys):                          # 5. check_frequency
    rfreq = np.array([1e-20, 1, 1e06])
    retaH = np.array([[0.05 + 5.56325028e-30j, 50 + 2.78162514e-30j],
                      [0.05 + 5.56325028e-10j, 50 + 2.78162514e-10j],
                      [0.05 + 5.56325028e-04j, 50 + 2.78162514e-04j]])
    retaV = np.array([[0.05 + 1.11265006e-29j, 5.55555556 + 2.78162514e-29j],
                      [0.05 + 1.11265006e-09j, 5.55555556 + 2.78162514e-09j],
                      [0.05 + 1.11265006e-03j, 5.55555556 + 2.78162514e-03j]])
    rzetaH = np.array([[0 + 7.89568352e-26j, 0 + 7.89568352e-26j],
                       [0 + 7.89568352e-06j, 0 + 7.89568352e-06j],
                       [0 + 7.89568352e+00j, 0 + 7.89568352e+00j]])
    rzetaV = np.array([[0 + 7.89568352e-25j, 0 + 3.94784176e-25j],
                       [0 + 7.89568352e-05j, 0 + 3.94784176e-05j],
                       [0 + 7.89568352e+01j, 0 + 3.94784176e+01j]])

    output = utils.check_frequency(np.array([0, 1, 1e6]), np.array([20, .02]),
                                   np.array([1, 3]), np.array([10, 5]),
                                   np.array([20, 50]), np.array([1, 1]),
                                   np.array([10, 5]), 3)
    out, _ = capsys.readouterr()
    assert out[:27] == "* WARNING :: Frequencies < "
    freq, etaH, etaV, zetaH, zetaV = output
    assert_allclose(freq, rfreq)
    assert_allclose(etaH, retaH)
    assert_allclose(etaV, retaV)
    assert_allclose(zetaH, rzetaH)
    assert_allclose(zetaV, rzetaV)


# 6. check_hankel

# 7. check_model

# 8. check_opt

# 9. check_time


def test_get_abs(capsys):                                         # 10. get_abs
    # Check some cases
    #       general,  x/y-pl,  x/z-pl    x
    ang = [[45, 45], [30, 0], [0, 60], [0, 0]]
    # Results for EE, ME, EM, MM
    res = [[11, 12, 13, 21, 22, 23, 31, 32, 33], [11, 12, 13, 21, 22, 23],
           [11, 12, 13, 31, 32, 33], [11, 12, 13], [11, 12, 21, 22, 31, 32],
           [11, 12, 21, 22], [11, 12, 31, 32], [11, 12],
           [11, 13, 21, 23, 31, 33], [11, 13, 21, 23], [11, 13, 31, 33],
           [11, 13], [11, 21, 31], [11, 21], [11, 31], [11],
           [14, 24, 34, 15, 25, 35, 16, 26, 36], [14, 24, 34, 15, 25, 35],
           [14, 24, 34, 16, 26, 36], [14, 24, 34], [14, 24, 15, 25, 16, 26],
           [14, 24, 15, 25], [14, 24, 16, 26], [14, 24],
           [14, 34, 15, 35, 16, 36], [14, 34, 15, 35], [14, 34, 16, 36],
           [14, 34], [14, 15, 16], [14, 15], [14, 16], [14],
           [14, 15, 16, 24, 25, 26, 34, 35, 36], [14, 15, 16, 24, 25, 26],
           [14, 15, 16, 34, 35, 36], [14, 15, 16], [14, 15, 24, 25, 34, 35],
           [14, 15, 24, 25], [14, 15, 34, 35], [14, 15],
           [14, 16, 24, 26, 34, 36], [14, 16, 24, 26], [14, 16, 34, 36],
           [14, 16], [14, 24, 34], [14, 24], [14, 34], [14],
           [11, 12, 13, 21, 22, 23, 31, 32, 33], [11, 12, 13, 21, 22, 23],
           [11, 12, 13, 31, 32, 33], [11, 12, 13], [11, 12, 21, 22, 31, 32],
           [11, 12, 21, 22], [11, 12, 31, 32], [11, 12],
           [11, 13, 21, 23, 31, 33], [11, 13, 21, 23], [11, 13, 31, 33],
           [11, 13], [11, 21, 31], [11, 21], [11, 31], [11]]
    i = 0
    for msrc in [False, True]:
        for mrec in [False, True]:
            for src in ang:
                for rec in ang:
                    out = utils.get_abs(msrc, mrec, src[0], src[1], rec[0],
                                        rec[1], 0)
                    assert_allclose(out, res[i])
                    i += 1

    # Check print statement
    _ = utils.get_abs(True, True, 90, 0, 0, 90, 3)
    out, _ = capsys.readouterr()
    assert out == "   Required ab's   :  11 12 31 32\n"

# 11. get_geo_fact


def test_get_layer_nr():                                     # 12. get_layer_nr
    bip = np.array([0, 0, 300])
    lbip, zbip = utils.get_layer_nr(bip, np.array([-np.infty, 500]))
    assert lbip == 0
    assert zbip == 300
    lbip, _ = utils.get_layer_nr(bip, np.array([-np.infty, 0, 300, 500]))
    assert lbip == 1
    lbip, _ = utils.get_layer_nr(bip, np.array([-np.infty, 0, 200]))
    assert lbip == 2
    bip = np.array([np.zeros(4), np.zeros(4), np.arange(4)*100])
    lbip, _ = utils.get_layer_nr(bip, np.array([-np.infty, 0, 200]))
    assert_allclose(lbip, [0, 1, 1, 2])


def test_get_off_ang(capsys):                                 # 13. get_off_ang
    src = [np.array([0, 100]), np.array([0, 100]), np.array([0, 100])]
    rec = [np.array([0, 5000]), np.array([0, 100]), np.array([0, 200])]
    resoff = np.array([0.001, 5001, 141.42135623730951, 4900])
    resang = np.array([np.nan, 0.019997333973150531, -2.3561944901923448, 0.])
    off, ang = utils.get_off_ang(src, rec, 2, 2, 3)
    out, _ = capsys.readouterr()
    assert out[:23] == "* WARNING :: Offsets < "
    assert_allclose(off, resoff)
    assert_allclose(ang, resang, equal_nan=True)


# 14. get_azm_dip


def test_printstartfinish(capsys):                       # 15. printstartfinish
    t0 = utils.printstartfinish(0)
    assert type(t0) == float
    out, _ = capsys.readouterr()
    assert out == ""

    t0 = utils.printstartfinish(3)
    assert type(t0) == float
    out, _ = capsys.readouterr()
    assert out == "\n:: empymod START  ::\n\n"

    utils.printstartfinish(0, t0)
    out, _ = capsys.readouterr()
    assert out == ""

    utils.printstartfinish(3, t0)
    out, _ = capsys.readouterr()
    assert out[:27] == "\n:: empymod END; runtime = "

    utils.printstartfinish(3, t0, 13)
    out, _ = capsys.readouterr()
    assert out[-19:] == "13 kernel call(s)\n\n"


def test_conv_warning(capsys):                               # 16. conv_warning
    # If converged, no output
    utils.conv_warning(True, ['', '', '', 51, ''], 'Hankel', 0)
    out, _ = capsys.readouterr()
    assert out == ""

    # If not converged, but verb=0, no output
    utils.conv_warning(False, ['', '', '', 51, ''], 'Hankel', 0)
    out, _ = capsys.readouterr()
    assert out == ""

    # If not converged, and verb>0, print
    utils.conv_warning(False, ['', '', '', 51, ''], 'Hankel', 1)
    out, _ = capsys.readouterr()
    assert out[:35] == "* WARNING :: Hankel-QWE used all 51"

    # If converged, and verb>1, no output
    utils.conv_warning(True, ['', '', '', 51, ''], 'Hankel', 1)
    out, _ = capsys.readouterr()
    assert out == ""


def test_check_shape(capsys):                                # 17. _check_shape
    # Ensure no Error is raised
    utils._check_shape(np.zeros((3, 4)), 'tvar', (3, 4))
    utils._check_shape(np.zeros((3, 4)), 'tvar', (3, 4), (2, ))
    utils._check_shape(np.zeros((3, 4)), 'tvar', (2,), (3, 4))
    # Ensure Error is raised
    with pytest.raises(ValueError):
        utils._check_shape(np.zeros((3, 4)), 'tvar', (2,))
    with pytest.raises(ValueError):
        utils._check_shape(np.zeros((3, 4)), 'tvar', (2,), (1, 4))


def test_check_var():                                          # 18. _check_var
    # This is basically np.array(), with an optional call to _check_shape
    # above. Just three simple checks therefore; one without call, one with one
    # shape, and one with two shapes

    # Without shapes
    out = utils._check_var(np.pi, int, 3, 'tvar')
    assert out[0, 0, 0] == 3

    # One shape, but wrong
    with pytest.raises(ValueError):
        out = utils._check_var(np.arange(3)*.5, float, 1, 'tvar', (1, 3))

    # Two shapes, second one is correct
    out = utils._check_var(np.arange(3)*.5, float, 1, 'tvar', (1, 3), (3, ))


def test_strvar():                                                # 19. _strvar
    out = utils._strvar(np.arange(3)*np.pi)
    assert out == "0 3.14159 6.28319"

    out = utils._strvar(np.pi, '{:20.10e}')
    assert out == "    3.1415926536e+00"


def test_prnt_min_max_val(capsys):                      # 20. _prnt_min_max_val
    utils._prnt_min_max_val(np.arange(1, 5)*2, 'tvar', 0)
    out, _ = capsys.readouterr()
    assert out == "tvar 2 - 8 : 4  [min-max; #]\n"

    utils._prnt_min_max_val(np.arange(1, 5)*2, 'tvar', 4)
    out, _ = capsys.readouterr()
    outstr = "tvar 2 - 8 : 4  [min-max; #]\n                   :  2 4 6 8\n"
    assert out == outstr

    utils._prnt_min_max_val(np.array(1), 'tvar', 0)
    out, _ = capsys.readouterr()
    assert out == "tvar 1\n"

    utils._prnt_min_max_val(np.array(1), 'tvar', 4)
    out, _ = capsys.readouterr()
    assert out == "tvar 1\n"


def test_check_min(capsys):                                    # 21. _check_min
    # Have to provide copies, as they are changed in place...good/bad?
    # inp > minval verb = 0
    inp1 = np.array([1e-3])
    out1 = utils._check_min(inp1.copy(), 1e-20, 'name', 'unit', 0)
    out, _ = capsys.readouterr()
    assert out == ""
    assert_allclose(inp1, out1)
    # inp > minval verb = 1
    inp2 = np.array([1e3, 10])
    out2 = utils._check_min(inp2.copy(), 1e-10, 'name', 'unit', 1)
    out, _ = capsys.readouterr()
    assert out == ""
    assert_allclose(inp2, out2)
    # inp < minval verb = 0
    inp3 = np.array([1e-6])
    out3 = utils._check_min(inp3.copy(), 1e-3, 'name', 'unit', 0)
    out, _ = capsys.readouterr()
    assert out == ""
    assert_allclose(np.array([1e-3]), out3)
    # inp < minval verb = 1
    inp4 = np.array([1e-20, 1e-3])
    out4 = utils._check_min(inp4.copy(), 1e-15, 'name', 'unit', 1)
    out, _ = capsys.readouterr()
    assert out[:35] == "* WARNING :: name < 1e-15 unit are "
    assert_allclose(np.array([1e-15, 1e-3]), out4)
