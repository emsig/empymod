import sys
import pytest
import scooby
import subprocess
import numpy as np
from numpy.testing import assert_allclose

from empymod import utils, filters


def test_emarray():
    out = utils.EMArray(3)
    assert out.amp() == 3
    assert out.pha() == 0
    assert out.real == 3
    assert out.imag == 0

    out = utils.EMArray(1+1j)
    assert out.amp() == np.sqrt(2)
    assert_allclose(out.pha(), np.pi/4)
    assert out.real == 1
    assert out.imag == 1

    out = utils.EMArray([1+1j, 0+1j, -1-1j])
    assert_allclose(out.amp(), [np.sqrt(2), 1, np.sqrt(2)])
    assert_allclose(out.pha(unwrap=False), [np.pi/4, np.pi/2, -3*np.pi/4])
    assert_allclose(out.pha(deg=True, unwrap=False), [45., 90., -135.])
    assert_allclose(out.pha(deg=True, unwrap=False, lag=False),
                    [-45., -90., 135.])
    assert_allclose(out.pha(deg=True, lag=False), [-45., -90., -225.])
    assert_allclose(out.real, [1, 0, -1])
    assert_allclose(out.imag, [1, 1, -1])


def test_check_ab(capsys):
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
    for i, val in enumerate(iab):
        ab, msrc, mrec = utils.check_ab(val, 0)
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
    with pytest.raises(ValueError, match='<ab> must be one of: '):
        utils.check_ab(77, 0)

    # We just check one other thing here, that it fails with a TypeError if a
    # list instead of one value is provided. Generally the try/except statement
    # with int() should take proper care of all the checking right in check_ab.
    with pytest.raises(TypeError, match='<ab> must be an integer'):
        utils.check_ab([12, ], 0)


def test_check_bipole():
    # Wrong size
    with pytest.raises(ValueError, match='Parameter tvar has wrong length!'):
        utils.check_bipole([0, 0, 0], 'tvar')

    # # Dipole stuff

    # Normal case
    ipole = [[0, 0, 0], [10, 20, 30], [100, 0, 100], 0, 32]
    inp_type = type(ipole[0])
    pole, nout, outz, isdipole = utils.check_bipole(ipole, 'tvar')
    out_type = type(ipole[0])
    assert inp_type == out_type  # Check input wasn't altered.
    assert_allclose(pole[0], np.array([0, 0, 0]))
    assert_allclose(pole[1], np.array([10, 20, 30]))
    assert_allclose(pole[2], np.array([100, 0, 100]))
    assert nout == 3
    assert outz == 3
    assert_allclose(isdipole, True)

    # Multiple azimuth
    pole = [[0, 0, 0], [10, 20, 30], [100, 0, 100], [0, 1, 2], 1]
    assert_allclose(pole[0], np.array([0, 0, 0]))
    assert_allclose(pole[1], np.array([10, 20, 30]))
    assert_allclose(pole[2], np.array([100, 0, 100]))
    assert_allclose(pole[3], np.array([0, 1, 2]))
    assert_allclose(pole[4], np.array([1, 1, 1]))

    # Multiple dip
    pole = [[0, 0, 0], [10, 20, 30], [100, 0, 100], 1, [0, 1, 2]]
    assert_allclose(pole[0], np.array([0, 0, 0]))
    assert_allclose(pole[1], np.array([10, 20, 30]))
    assert_allclose(pole[2], np.array([100, 0, 100]))
    assert_allclose(pole[3], np.array([1, 1, 1]))
    assert_allclose(pole[4], np.array([0, 1, 2]))

    # x.size != y.size
    pole = [[0, 0], [10, 20, 30], [100, 0, 100], 0, 0]
    with pytest.raises(ValueError, match='Parameter tvar-y has wrong shape'):
        utils.check_bipole(pole, 'tvar')

    # # Bipole stuff

    # Dipole instead bipole
    pole = [0, 0, 1000, 1000, 10, 10]
    with pytest.raises(ValueError, match='At least one of <tvar> is a point'):
        utils.check_bipole(pole, 'tvar')

    # Normal case
    ipole = [0, 0, 1000, 1000, 10, 20]
    inp_type = type(ipole[0])
    pole, nout, outz, isdipole = utils.check_bipole(ipole, 'tvar')
    out_type = type(ipole[0])
    assert inp_type == out_type  # Check input wasn't altered.
    assert_allclose(pole[0], 0)
    assert_allclose(pole[1], 0)
    assert_allclose(pole[2], 1000)
    assert_allclose(pole[3], 1000)
    assert_allclose(pole[4], 10)
    assert_allclose(pole[5], 20)
    assert nout == 1
    assert outz == 1
    assert_allclose(isdipole, False)

    # Pole one has variable depths
    pole = [[0, 0], [10, 10], [0, 0], [20, 30], [10, 20], 0]
    pole, nout, outz, _ = utils.check_bipole(pole, 'tvar')
    assert_allclose(pole[4], [10, 20])
    assert_allclose(pole[5], [0, 0])
    assert nout == 2
    assert outz == 2

    # Pole one has variable depths
    pole = [[0, 0], [10, 10], [0, 0], [20, 30], 10, [20, 0]]
    pole, nout, outz, _ = utils.check_bipole(pole, 'tvar')
    assert_allclose(pole[4], [10, 10])
    assert_allclose(pole[5], [20, 0])
    assert nout == 2
    assert outz == 2


def test_check_dipole(capsys):
    # correct input, verb > 2, src
    isrc = [[1000, 2000], [0, 0], 0]
    inp_type = type(isrc[0])
    src, nsrc = utils.check_dipole(isrc, 'src', 3)
    out, _ = capsys.readouterr()
    out_type = type(isrc[0])
    assert inp_type == out_type  # Check input wasn't altered.
    assert nsrc == 2
    assert_allclose(src[0], [1000, 2000])
    assert_allclose(src[1], [0, 0])
    assert_allclose(src[2], 0)

    outstr = "   Source(s)       :  2 dipole(s)\n"
    outstr += "     > x       [m] :  1000 2000\n"
    outstr += "     > y       [m] :  0 0\n"
    outstr += "     > z       [m] :  0\n"
    assert out == outstr

    # Check print if more than 3 dipoles
    utils.check_dipole([[1, 2, 3, 4], [0, 0, 0, 0], 0], 'src', 4)
    out, _ = capsys.readouterr()
    outstr = "   Source(s)       :  4 dipole(s)\n"
    outstr += "     > x       [m] :  1 - 4 : 4  [min-max; #]\n"
    outstr += "                   :  1 2 3 4\n"
    outstr += "     > y       [m] :  0 - 0 : 4  [min-max; #]\n"
    outstr += "                   :  0 0 0 0\n"
    outstr += "     > z       [m] :  0\n"
    assert out == outstr

    # correct input, verb > 2, rec
    rec, nrec = utils.check_dipole([0, 0, 0], 'rec', 3)
    out, _ = capsys.readouterr()
    assert nrec == 1
    assert_allclose(rec[0], 0)
    assert_allclose(rec[1], 0)
    assert_allclose(rec[2], 0)

    outstr = "   Receiver(s)     :  1 dipole(s)\n"
    outstr += "     > x       [m] :  0\n"
    outstr += "     > y       [m] :  0\n"
    outstr += "     > z       [m] :  0\n"
    assert out == outstr

    # Check expansion
    rec, nrec = utils.check_dipole([[0, 1], 0, 0], 'rec', 0)
    assert_allclose(rec[1], [0, 0])
    rec, nrec = utils.check_dipole([3, [0, 1, 2], 0], 'rec', 0)
    assert_allclose(rec[0], [3, 3, 3])

    # Check Errors: more than one z
    with pytest.raises(ValueError, match='Parameter src has wrong shape'):
        utils.check_dipole([[0, 0], [0, 0], [0, 0]], 'src', 3)
    # Check Errors: wrong number of elements
    with pytest.raises(ValueError, match='Parameter rec has wrong shape'):
        utils.check_dipole([0, 0, 0, 0], 'rec', 3)
    # Check Errors: x- and y- of different sizes, != 1
    with pytest.raises(ValueError, match='Parameters rec-x and rec-y must be'):
        utils.check_dipole([[0, 1], [0, 1, 2], 0], 'rec', 3)


def test_check_frequency(capsys):
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
    assert "   frequency  [Hz] :  " in out
    assert "* WARNING :: Frequencies < " in out
    freq, etaH, etaV, zetaH, zetaV = output
    assert_allclose(freq, rfreq)
    assert_allclose(etaH, retaH)
    assert_allclose(etaV, retaV)
    assert_allclose(zetaH, rzetaH)
    assert_allclose(zetaV, rzetaV)

    output = utils.check_frequency(-np.array([1e-40, 1, 1e6]),
                                   np.array([20, .02]),
                                   np.array([1, 3]), np.array([10, 5]),
                                   np.array([20, 50]), np.array([1, 1]),
                                   np.array([10, 5]), 3)
    out, _ = capsys.readouterr()
    assert "   s-value    [Hz] :  " in out
    assert "* WARNING :: Laplace val < " in out
    freq, etaH, etaV, zetaH, zetaV = output
    assert_allclose(freq, rfreq)


def test_check_hankel(capsys):
    # # DLF # #
    # verbose
    ht, htarg = utils.check_hankel('dlf', {}, 4)
    out, _ = capsys.readouterr()
    assert "   Hankel          :  DLF (Fast Hankel Transform)\n     > F" in out
    assert "     > DLF type    :  Standard" in out
    assert ht == 'dlf'
    assert htarg['dlf'].name == filters.Hankel().key_201_2009.name
    assert htarg['pts_per_dec'] == 0

    # provide filter-string and unknown parameter
    _, htarg = utils.check_hankel('dlf', {'dlf': 'key_201_2009', 'abc': 0}, 1)
    out, _ = capsys.readouterr()
    assert htarg['dlf'].name == filters.Hankel().key_201_2009.name
    assert htarg['pts_per_dec'] == 0
    assert "WARNING :: Unknown htarg {'abc': 0} for method 'dlf'" in out

    # provide filter-instance
    _, htarg = utils.check_hankel(
            'dlf', {'dlf': filters.Hankel().kong_61_2007b}, 0)
    assert htarg['dlf'].name == filters.Hankel().kong_61_2007b.name
    assert htarg['pts_per_dec'] == 0

    # provide pts_per_dec
    _, htarg = utils.check_hankel('dlf', {'pts_per_dec': -1}, 3)
    out, _ = capsys.readouterr()
    assert "     > DLF type    :  Lagged Convolution" in out
    assert htarg['dlf'].name == filters.Hankel().key_201_2009.name
    assert htarg['pts_per_dec'] == -1

    # provide filter-string and pts_per_dec
    _, htarg = utils.check_hankel(
            'dlf', {'dlf': 'key_201_2009', 'pts_per_dec': 20}, 4)
    out, _ = capsys.readouterr()
    assert "     > DLF type    :  Splined, 20.0 pts/dec" in out
    assert htarg['dlf'].name == filters.Hankel().key_201_2009.name
    assert htarg['pts_per_dec'] == 20

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    ht, htarg = utils.check_hankel('dlf', {}, 1)
    _, _ = utils.check_hankel(ht, htarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # # QWE # #
    # verbose
    ht, htarg = utils.check_hankel('qwe', {}, 4)
    out, _ = capsys.readouterr()
    outstr = "   Hankel          :  Quadrature-with-Extrapolation\n     > rtol"
    assert outstr in out
    assert ht == 'qwe'
    assert htarg['rtol'] == 1e-12
    assert htarg['atol'] == 1e-30
    assert htarg['nquad'] == 51
    assert htarg['maxint'] == 100
    assert htarg['pts_per_dec'] == 0
    assert htarg['diff_quad'] == 100
    assert htarg['a'] is None
    assert htarg['b'] is None
    assert htarg['limit'] is None

    # limit
    _, htarg = utils.check_hankel('qwe', {'limit': 30}, 0)
    assert htarg['rtol'] == 1e-12
    assert htarg['atol'] == 1e-30
    assert htarg['nquad'] == 51
    assert htarg['maxint'] == 100
    assert htarg['pts_per_dec'] == 0
    assert htarg['diff_quad'] == 100
    assert htarg['a'] is None
    assert htarg['b'] is None
    assert htarg['limit'] == 30

    # all arguments
    _, htarg = utils.check_hankel(
            'qwe', {'rtol': 1e-3, 'atol': 1e-4, 'nquad': 31, 'maxint': 20,
                    'pts_per_dec': 30, 'diff_quad': 200, 'a': 1e-6, 'b': 160,
                    'limit': 30},
            3)
    out, _ = capsys.readouterr()
    assert "     > a     (quad):  1e-06" in out
    assert "     > b     (quad):  160" in out
    assert "     > limit (quad):  30" in out
    assert htarg['rtol'] == 1e-3
    assert htarg['atol'] == 1e-4
    assert htarg['nquad'] == 31
    assert htarg['maxint'] == 20
    assert htarg['pts_per_dec'] == 30
    assert htarg['diff_quad'] == 200
    assert htarg['a'] == 1e-6
    assert htarg['b'] == 160
    assert htarg['limit'] == 30

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    ht, htarg = utils.check_hankel('qwe', {}, 1)
    _, _ = utils.check_hankel(ht, htarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # # QUAD # #
    # verbose
    ht, htarg = utils.check_hankel('quad', {}, 4)
    out, _ = capsys.readouterr()
    outstr = "   Hankel          :  Quadrature\n     > rtol"
    assert outstr in out
    assert ht == 'quad'
    assert htarg['rtol'] == 1e-12
    assert htarg['atol'] == 1e-20
    assert htarg['limit'] == 500
    assert htarg['a'] == 1e-6
    assert htarg['b'] == 0.1
    assert htarg['pts_per_dec'] == 40

    # pts_per_dec
    _, htarg = utils.check_hankel('quad', {'pts_per_dec': 100}, 0)
    assert htarg['rtol'] == 1e-12
    assert htarg['atol'] == 1e-20
    assert htarg['limit'] == 500
    assert htarg['a'] == 1e-6
    assert htarg['b'] == 0.1
    assert htarg['pts_per_dec'] == 100
    # all arguments
    _, htarg = utils.check_hankel(
            'quad', {'rtol': 1e-3, 'atol': 1e-4, 'limit': 100, 'a': 1e-10,
                     'b': 200, 'pts_per_dec': 50},
            0)
    assert htarg['rtol'] == 1e-3
    assert htarg['atol'] == 1e-4
    assert htarg['limit'] == 100
    assert htarg['a'] == 1e-10
    assert htarg['b'] == 200
    assert htarg['pts_per_dec'] == 50

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    ht, htarg = utils.check_hankel('quad', {}, 1)
    _, _ = utils.check_hankel(ht, htarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # wrong ht
    with pytest.raises(ValueError, match='must be one of: '):
        utils.check_hankel('doesnotexist', {}, 1)

    # filter missing attributes
    dlf = filters.DigitalFilter('test')
    with pytest.raises(AttributeError, match='DLF-filter is missing some'):
        utils.check_hankel('dlf', {'dlf': dlf}, 1)


def test_check_model(capsys):
    # Normal case; xdirect=True (default)
    res = utils.check_model(0, [1e20, 20], [1, 0], [0, 1], [50, 80], [10, 1],
                            [1, 1], True, 3)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = res
    out, _ = capsys.readouterr()
    assert "* WARNING :: Parameter aniso < " in out
    assert "   direct field    :  Comp. in frequency domain" in out
    assert_allclose(depth, [-np.inf, 0])
    assert_allclose(res, [1e20, 20])
    assert_allclose(aniso, [1, np.sqrt(1e-20/20)])
    assert_allclose(epermH, [0, 1])
    assert_allclose(epermV, [50, 80])
    assert_allclose(mpermH, [10, 1])
    assert_allclose(mpermV, [1, 1])
    assert_allclose(isfullspace, False)

    # xdirect=False
    res = utils.check_model(0, [1e20, 20], [1, 2], [0, 1], [50, 80], [10, 1],
                            [1, 1], False, 3)
    out, _ = capsys.readouterr()
    assert "   direct field    :  Comp. in wavenumber domain" in out

    # xdirect=None
    res = utils.check_model(0, [1e20, 20], [1, 2], [0, 1], [50, 80], [10, 1],
                            [1, 1], None, 3)
    out, _ = capsys.readouterr()
    assert "   direct field    :  Not calculated (secondary field)" in out

    # Check -np.inf is added to depth
    out = utils.check_model([], 2, 1, 1, 1, 1, 1, True, 1)
    assert_allclose(out[0], -np.inf)

    # Check -np.inf is not added if it is already in depth
    out = utils.check_model(-np.inf, 2, 1, 1, 1, 1, 1, True, 1)
    assert_allclose(out[0], -np.inf)

    # Check verbosity and fullspace
    utils.check_model(0, [1, 1], [2, 2], [10, 10], [1, 1], None, [3, 3], True,
                      4)
    out, _ = capsys.readouterr()
    outstr1 = "   depth       [m] :  0\n   res     [Ohm.m] :  1 1\n   aniso"
    outstr2 = "S A FULLSPACE; returning analytical frequency-domain solution\n"
    assert outstr1 in out
    assert outstr2 in out

    # Check fullspace if only one value, w\o xdirect
    utils.check_model([], 1, 2, 10, 1, 2, 3, True, 4)
    out, _ = capsys.readouterr()
    assert outstr2 in out
    utils.check_model([], 1, 2, 10, 1, 2, 3, False, 4)
    out, _ = capsys.readouterr()
    assert "MODEL IS A FULLSPACE\n" in out

    # Non-continuously in/de-creasing depth
    with pytest.raises(ValueError, match='Depth must be continuously incr'):
        var = [1, 1, 1, 1]
        utils.check_model([0, 100, 90], var, var, var, var, var, var, True, 1)

    # A ValueError check
    with pytest.raises(ValueError, match='Parameter res has wrong shape'):
        utils.check_model(
                0, 1, [2, 2], [10, 10], [1, 1], [2, 2], [3, 3], True, 1)


def test_check_all_depths():
    depth = np.array([-50, 0, 100, 2000])
    res = [6, 1, 2, 3, 4]
    aniso = [6, 7, 8, 9, 10]
    epermH = [1.0, 1.1, 1.2, 1.3, 1.4]
    epermV = [1.5, 1.6, 1.7, 1.8, 1.9]
    mpermH = [2.0, 2.1, 2.2, 2.3, 2.4]
    mpermV = [2.5, 2.6, 2.7, 2.8, 2.9]

    # 1. Ordering as internally used:

    # LHS low-to-high (+1, ::+1)
    lhs_l2h = utils.check_model(
            depth, res, aniso, epermH, epermV, mpermH, mpermV, True, 0)

    # RHS high-to-low (-1, ::+1)
    rhs_h2l = utils.check_model(
            -depth, res, aniso, epermH, epermV, mpermH, mpermV, True, 0)

    # 2. Reversed ordering:

    # LHS high-to-low (+1, ::-1)
    lhs_h2l = utils.check_model(
            depth[::-1], res[::-1], aniso[::-1], epermH[::-1], epermV[::-1],
            mpermH[::-1], mpermV[::-1], True, 0)

    # RHS low-to-high (-1, ::-1)
    rhs_l2h = utils.check_model(
            -depth[::-1], res[::-1], aniso[::-1], epermH[::-1], epermV[::-1],
            mpermH[::-1], mpermV[::-1], True, 0)

    for i, var in enumerate([lhs_h2l, rhs_h2l, rhs_l2h]):
        if i == 0:
            swap = 1   # LHS
        else:
            swap = -1  # RHS
        assert_allclose(lhs_l2h[0][1:], swap*var[0][1:][::swap])
        assert_allclose(lhs_l2h[1], var[1][::swap])
        assert_allclose(lhs_l2h[2], var[2][::swap])
        assert_allclose(lhs_l2h[3], var[3][::swap])
        assert_allclose(lhs_l2h[4], var[4][::swap])
        assert_allclose(lhs_l2h[5], var[5][::swap])
        assert_allclose(lhs_l2h[6], var[6][::swap])


def test_check_time(capsys):
    time = np.array([3])

    # # DLF # #
    # verbose
    _, f, ft, ftarg = utils.check_time(time, 0, 'dlf', {}, 4)
    out, _ = capsys.readouterr()
    assert "   time        [s] :  3" in out
    assert "   Fourier         :  DLF (Sine-Filter)" in out
    assert "> DLF type    :  Lagged Convolution" in out
    assert ft == 'dlf'
    assert ftarg['dlf'].name == filters.Fourier().key_201_2012.name
    assert ftarg['pts_per_dec'] == -1
    f1 = np.array([4.87534752e-08, 5.60237934e-08, 6.43782911e-08,
                   7.39786458e-08, 8.50106448e-08, 9.76877807e-08,
                   1.12255383e-07, 1.28995366e-07, 1.48231684e-07])
    f2 = np.array([2.88109455e+04, 3.31073518e+04, 3.80444558e+04,
                   4.37178011e+04, 5.02371788e+04, 5.77287529e+04,
                   6.63375012e+04, 7.62300213e+04, 8.75977547e+04])
    assert_allclose(f[:9], f1)
    assert_allclose(f[-9:], f2)
    assert_allclose(f.size, 201+3)
    assert ftarg['kind'] == 'sin'

    # filter-string and unknown parameter
    _, f, _, ftarg = utils.check_time(
            time, -1, 'dlf',
            {'dlf': 'key_201_2012', 'kind': 'cos', 'notused': 1},
            4)
    out, _ = capsys.readouterr()
    outstr = "   time        [s] :  3\n"
    outstr += "   Fourier         :  DLF (Cosine-Filter)\n     > Filter"
    assert outstr in out
    assert "WARNING :: Unknown ftarg {'notused': 1} for method 'dlf'" in out
    assert ft == 'dlf'
    assert ftarg['dlf'].name == filters.Fourier().key_201_2012.name
    assert ftarg['pts_per_dec'] == -1
    assert_allclose(f[:9], f1)
    assert_allclose(f[-9:], f2)
    assert_allclose(f.size, 201+3)
    assert ftarg['kind'] == 'cos'

    # filter instance
    _, _, _, ftarg = utils.check_time(
            time, 1, 'dlf',
            {'dlf': filters.Fourier().key_201_2012, 'kind': 'sin'}, 0)
    assert ftarg['dlf'].name == filters.Fourier().key_201_2012.name
    assert ftarg['pts_per_dec'] == -1
    assert ftarg['kind'] == 'sin'

    # pts_per_dec
    out, _ = capsys.readouterr()  # clear buffer
    _, _, _, ftarg = utils.check_time(time, 0, 'dlf', {'pts_per_dec': 30}, 4)
    assert ftarg['dlf'].name == filters.Fourier().key_201_2012.name
    assert ftarg['pts_per_dec'] == 30
    assert ftarg['kind'] == 'sin'
    out, _ = capsys.readouterr()
    assert "     > DLF type    :  Splined, 30.0 pts/dec" in out

    # filter-string and pts_per_dec
    _, _, _, ftarg = utils.check_time(
            time, 0, 'dlf',
            {'dlf': 'key_81_2009', 'pts_per_dec': -1, 'kind': 'cos'}, 4)
    out, _ = capsys.readouterr()
    assert "     > DLF type    :  Lagged Convolution" in out
    assert ftarg['dlf'].name == filters.Fourier().key_81_2009.name
    assert ftarg['pts_per_dec'] == -1
    assert ftarg['kind'] == 'cos'

    # pts_per_dec
    _, freq, _, ftarg = utils.check_time(
            time, 0, 'dlf', {'pts_per_dec': 0, 'kind': 'sin'}, 4)
    out, _ = capsys.readouterr()
    assert "     > DLF type    :  Standard" in out
    assert ftarg['pts_per_dec'] == 0
    f_base = filters.Fourier().key_201_2012.base
    assert_allclose(np.ravel(f_base/(2*np.pi*time[:, None])), freq)

    # filter-string and pts_per_dec
    _, _, _, ftarg = utils.check_time(
            time, 0, 'dlf',
            {'dlf': 'key_81_2009', 'pts_per_dec': 50, 'kind': 'cos'}, 0)
    assert ftarg['dlf'].name == filters.Fourier().key_81_2009.name
    assert ftarg['pts_per_dec'] == 50
    assert ftarg['kind'] == 'cos'

    # just kind
    _, f, _, ftarg = utils.check_time(
            time, 0, 'dlf', {'kind': 'sin'}, 0)
    assert ftarg['pts_per_dec'] == -1
    assert_allclose(f[:9], f1)
    assert_allclose(f[-9:], f2)
    assert_allclose(f.size, 204)

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    _, _, _, ftarg = utils.check_time(time, 0, 'dlf', {}, 1)
    _, _, _, _ = utils.check_time(time, 0, 'dlf', ftarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # # QWE # #
    # verbose
    _, f, ft, ftarg = utils.check_time(time, 0, 'qwe', {}, 4)
    out, _ = capsys.readouterr()
    outstr = "   Fourier         :  Quadrature-with-Extrapolation\n     > rtol"
    assert out[24:87] == outstr
    assert ft == 'qwe'
    assert ftarg['rtol'] == 1e-8
    assert ftarg['atol'] == 1e-20
    assert ftarg['nquad'] == 21
    assert ftarg['maxint'] == 200
    assert ftarg['pts_per_dec'] == 20
    assert ftarg['diff_quad'] == 100
    f1 = np.array([3.16227766e-03, 3.54813389e-03, 3.98107171e-03,
                   4.46683592e-03, 5.01187234e-03, 5.62341325e-03,
                   6.30957344e-03, 7.07945784e-03, 7.94328235e-03])
    f2 = np.array([1.00000000e+02, 1.12201845e+02, 1.25892541e+02,
                   1.41253754e+02, 1.58489319e+02, 1.77827941e+02,
                   1.99526231e+02, 2.23872114e+02, 2.51188643e+02])
    assert_allclose(f[:9], f1)
    assert_allclose(f[-9:], f2)
    assert_allclose(f.size, 99)
    assert ftarg['a'] is None
    assert ftarg['b'] is None
    assert ftarg['limit'] is None
    assert ftarg['sincos'] is np.sin

    # only limit
    _, _, _, ftarg = utils.check_time(time, 1, 'qwe', {'limit': 30}, 0)
    assert ftarg['rtol'] == 1e-8
    assert ftarg['atol'] == 1e-20
    assert ftarg['nquad'] == 21
    assert ftarg['maxint'] == 200
    assert ftarg['pts_per_dec'] == 20
    assert ftarg['diff_quad'] == 100
    assert ftarg['a'] is None
    assert ftarg['b'] is None
    assert ftarg['limit'] == 30
    assert ftarg['sincos'] is np.sin

    # all arguments
    _, _, _, ftarg = utils.check_time(
            time, -1, 'qwe',
            {'rtol': 1e-3, 'atol': 1e-4, 'nquad': 31, 'maxint': 20,
             'pts_per_dec': 30, 'diff_quad': 200, 'a': 0.01, 'b': 0.2,
             'limit': 100},
            3)
    out, _ = capsys.readouterr()
    assert "     > a     (quad):  0.01" in out
    assert "     > b     (quad):  0.2" in out
    assert "     > limit (quad):  100" in out
    assert ftarg['rtol'] == 1e-3
    assert ftarg['atol'] == 1e-4
    assert ftarg['nquad'] == 31
    assert ftarg['maxint'] == 20
    assert ftarg['pts_per_dec'] == 30
    assert ftarg['diff_quad'] == 200
    assert ftarg['a'] == 0.01
    assert ftarg['b'] == 0.2
    assert ftarg['limit'] == 100
    assert ftarg['sincos'] is np.cos

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    _, _, _, ftarg = utils.check_time(time, 0, 'qwe', {}, 1)
    _, _, _, _ = utils.check_time(time, 0, 'qwe', ftarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # # FFTLog # #
    # verbose
    _, f, ft, ftarg = utils.check_time(time, 0, 'fftlog', {}, 4)
    out, _ = capsys.readouterr()
    outstr = "   Fourier         :  FFTLog\n     > pts_per_dec"
    assert outstr in out
    assert ft == 'fftlog'
    assert ftarg['pts_per_dec'] == 10
    assert_allclose(ftarg['add_dec'], np.array([-2.,  1.]))
    assert ftarg['q'] == 0
    tres = np.array([0.3571562, 0.44963302, 0.56605443, 0.71262031, 0.89713582,
                     1.12942708, 1.42186445, 1.79002129, 2.25350329,
                     2.83699255, 3.57156202, 4.49633019, 5.66054433,
                     7.1262031, 8.97135818, 11.29427079, 14.2186445,
                     17.90021288, 22.53503287, 28.36992554, 35.71562019,
                     44.96330186, 56.60544331, 71.26203102, 89.71358175,
                     112.94270785, 142.18644499, 179.00212881, 225.35032873,
                     283.69925539])
    assert ftarg['mu'] == 0.5
    assert_allclose(ftarg['tcalc'], tres)
    assert_allclose(ftarg['dlnr'], 0.23025850929940461)
    assert_allclose(ftarg['kr'], 1.0610526667295022)
    assert_allclose(ftarg['rk'], 0.016449035064149849)

    fres = np.array([0.00059525, 0.00074937, 0.00094341, 0.00118768, 0.0014952,
                     0.00188234, 0.00236973, 0.00298331, 0.00375577,
                     0.00472823, 0.00595249, 0.00749374, 0.00943407,
                     0.01187678, 0.01495199, 0.01882343, 0.0236973,
                     0.02983313, 0.03755769, 0.04728233, 0.05952493,
                     0.07493744, 0.09434065, 0.11876785, 0.14951986,
                     0.18823435, 0.23697301, 0.29833134, 0.3755769,
                     0.47282331])
    assert_allclose(f, fres, rtol=1e-5)

    # Several parameters
    _, _, _, ftarg = utils.check_time(
            time, -1, 'fftlog',
            {'pts_per_dec': 10, 'add_dec': [-3, 4], 'q': 2}, 0)
    assert ftarg['pts_per_dec'] == 10
    assert_allclose(ftarg['add_dec'], np.array([-3.,  4.]))
    assert ftarg['q'] == 1  # q > 1 reset to 1...
    assert ftarg['mu'] == -0.5
    assert_allclose(ftarg['dlnr'], 0.23025850929940461)
    assert_allclose(ftarg['kr'], 0.94312869748639161)
    assert_allclose(ftarg['rk'], 1.8505737940600746)

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    _, _, _, ftarg = utils.check_time(time, 0, 'fftlog', {}, 1)
    _, _, _, _ = utils.check_time(time, 0, 'fftlog', ftarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # # FFT # #
    # verbose
    _, f, ft, ftarg = utils.check_time(time, 0, 'fft', {}, 4)
    out, _ = capsys.readouterr()
    assert "Fourier         :  Fast Fourier Transform FFT\n     > dfreq" in out
    assert "     > pts_per_dec :  (linear)" in out
    assert ft == 'fft'
    assert ftarg['dfreq'] == 0.002
    assert ftarg['nfreq'] == 2048
    assert ftarg['ntot'] == 2048
    assert ftarg['pts_per_dec'] is None
    fres = np.array([0.002, 0.004, 0.006, 0.008, 0.01, 4.088, 4.09, 4.092,
                     4.094, 4.096])
    assert_allclose(f[:5], fres[:5])
    assert_allclose(f[-5:], fres[-5:])

    # Several parameters
    _, _, _, ftarg = utils.check_time(
            time, 0, 'fft', {'dfreq': 1e-3, 'nfreq': 2**15+1, 'ntot': 3}, 0)
    assert ftarg['dfreq'] == 0.001
    assert ftarg['nfreq'] == 2**15+1
    assert ftarg['ntot'] == 2**16

    # Several parameters; pts_per_dec
    _, f, _, ftarg = utils.check_time(time, 0, 'fft', {'pts_per_dec': 5}, 3)
    out, _ = capsys.readouterr()
    assert "     > pts_per_dec :  5" in out
    assert ftarg['dfreq'] == 0.002
    assert ftarg['nfreq'] == 2048
    assert ftarg['ntot'] == 2048
    assert ftarg['pts_per_dec'] == 5
    outf = np.array([2.00000000e-03, 3.22098066e-03, 5.18735822e-03,
                     8.35419026e-03, 1.34543426e-02, 2.16680888e-02,
                     3.48962474e-02, 5.62000691e-02, 9.05096680e-02,
                     1.45764945e-01, 2.34753035e-01, 3.78067493e-01,
                     6.08874043e-01, 9.80585759e-01, 1.57922389e+00,
                     2.54332480e+00, 4.09600000e+00])

    assert_allclose(f, outf)

    # Assert it can be called repetitively
    _, _ = capsys.readouterr()
    _, _, _, ftarg = utils.check_time(time, 0, 'fft', {}, 1)
    _, _, _, _ = utils.check_time(time, 0, 'fft', ftarg, 1)
    out, _ = capsys.readouterr()
    assert out == ""

    # # Various # #

    # minimum time
    _ = utils.check_time(
            0, 0, 'dlf', {'dlf': 'key_201_2012', 'kind': 'cos'}, 1)
    out, _ = capsys.readouterr()
    assert out[:21] == "* WARNING :: Times < "

    # Signal != -1, 0, 1
    with pytest.raises(ValueError, match='<signal> must be one of:'):
        utils.check_time(time, -2, 'dlf', {}, 0)

    # ft != cos, sin, dlf, qwe, fftlog,
    with pytest.raises(ValueError, match='<ft> must be one of:'):
        utils.check_time(time, 0, 'bla', {}, 0)

    # filter missing attributes
    dlf = filters.DigitalFilter('test')
    with pytest.raises(AttributeError, match='DLF-filter is missing some att'):
        utils.check_time(time, 0, 'dlf', {'dlf': dlf}, 1)

    # filter with wrong kind
    with pytest.raises(ValueError, match="'kind' must be either 'sin' or"):
        utils.check_time(time, 0, 'dlf', {'kind': 'wrongkind'}, 1)


def test_check_solution(capsys):
    # wrong solution
    with pytest.raises(ValueError, match='Solution must be one of'):
        utils.check_solution('hs', 1, 13, False, False)

    # wrong ab/msrc/mrec
    with pytest.raises(ValueError, match='Diffusive solution is only imple'):
        utils.check_solution('dhs', None, 11, True, False)

    # wrong domain
    with pytest.raises(ValueError, match='Full fullspace solution is only'):
        utils.check_solution('fs', 1, 21, True, True)


def test_get_abs(capsys):
    # Check some cases
    #       general,  x/y-pl,  x/z-pl    x
    ang = [[np.pi/4, np.pi/4], [np.pi/6, 0], [0, np.pi/3], [0, 0]]
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

    # Check some more
    #       y/z-plane,  z-dir
    ang = [[np.pi/2, 0], [0, np.pi/2]]
    # Results for EE, ME, EM, MM
    res = [[22], [32], [23], [33], [25], [26], [35], [36], [25], [35], [26],
           [36], [22], [32], [23], [33]]

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

    # Assure that for different, but aligned angles, they are not deleted.
    ab_calc = utils.get_abs(
            False, False, np.array([0., np.pi/2]), np.array([0., 0.]),
            np.array([0.]), np.array([0.]), 0)
    assert_allclose(ab_calc, [11, 12])
    ab_calc = utils.get_abs(
            False, False, np.array([0., 0.]), np.array([3*np.pi/2, np.pi]),
            np.array([0.]), np.array([0.]), 0)
    assert_allclose(ab_calc, [11, 13])
    ab_calc = utils.get_abs(
            False, False, np.array([0.]), np.array([0.]),
            np.array([0., np.pi/2]), np.array([0., 0.]), 0)
    assert_allclose(ab_calc, [11, 21])
    ab_calc = utils.get_abs(
            False, False, np.array([0.]), np.array([0.]),
            np.array([0., 0.]), np.array([3*np.pi/2, np.pi]), 0)
    assert_allclose(ab_calc, [11, 31])


def test_get_geo_fact():
    res = np.array([0.017051023225738, 0.020779123804907, -0.11077204227395,
                    -0.081155809427821, -0.098900024313067, 0.527229048585517,
                    -0.124497144079623, -0.151717673241039, 0.808796206796408])
    res2 = np.rot90(np.fliplr(res.reshape(3, -1))).ravel()

    # EE, MM
    ab = [11, 12, 13, 21, 22, 23, 31, 32, 33]
    i = 0
    for i in range(9):
        out = utils.get_geo_fact(ab[i], 13.45, 23.8, 124.3, 5.3, False, False)
        assert_allclose(out[0], res[i])
        out = utils.get_geo_fact(ab[i], 13.45, 23.8, 124.3, 5.3, True, True)
        assert_allclose(out[0], res[i])
        i += 1

    # ME, EM
    ab = [14, 15, 16, 24, 25, 26, 34, 35, 36]
    i = 0
    for i in range(9):
        out = utils.get_geo_fact(ab[i], 13.45, 23.8, 124.3, 5.3, False, True)
        assert_allclose(out[0], res2[i])
        out = utils.get_geo_fact(ab[i], 13.45, 23.8, 124.3, 5.3, True, False)
        assert_allclose(out[0], res[i])
        i += 1


def test_get_layer_nr():
    bip = np.array([0, 0, 300])
    lbip, zbip = utils.get_layer_nr(bip, np.array([-np.inf, 500]))
    assert lbip == 0
    assert zbip == 300
    lbip, _ = utils.get_layer_nr(bip, np.array([-np.inf, 0, 300, 500]))
    assert lbip == 1
    lbip, _ = utils.get_layer_nr(bip, np.array([-np.inf, 0, 200]))
    assert lbip == 2
    bip = np.array([np.zeros(4), np.zeros(4), np.arange(4)*100])
    lbip, _ = utils.get_layer_nr(bip, np.array([-np.inf, 0, 200]))
    assert_allclose(lbip, [0, 1, 1, 2])


def test_get_off_ang(capsys):
    src = [np.array([0, 100]), np.array([0, 100]), np.array([0, 100])]
    rec = [np.array([0, 5000]), np.array([0, 100]), np.array([0, 200])]
    resoff = np.array([0.001, 5001, 141.42135623730951, 4900])
    resang = np.array([0.0, 0.019997333973150531, -2.3561944901923448, 0.])
    off, ang = utils.get_off_ang(src, rec, 2, 2, 3)
    out, _ = capsys.readouterr()
    assert out[:23] == "* WARNING :: Offsets < "
    assert_allclose(off, resoff)
    assert_allclose(ang, resang, equal_nan=True)


def test_get_azm_dip(capsys):
    # Dipole, src, ninpz = 1
    inp = [np.array([0]), np.array([0]), np.array([0]), np.array([0]),
           np.array([np.pi/4])]
    out = utils.get_azm_dip(inp, 0, 1, 1, True, 300, 'src', 0)
    assert out[0][0] == inp[0]
    assert out[0][1] == inp[1]
    assert out[0][2] == inp[2]
    assert out[0][3] == inp[3]
    assert out[0][4] == inp[4]
    assert out[1] == 0
    assert_allclose(out[2], 0.013707783890402)
    assert out[3] == 1
    assert out[4] == 1
    assert out[5] == 300

    # Dipole, rec, ninpz = 2, verbose
    inp = [np.array([0, 0]), np.array([0, 0]), np.array([0, 100]),
           np.array([np.pi/2]), np.array([np.pi/3])]
    out = utils.get_azm_dip(inp, 0, 2, 52, True, 300, 'rec', 4)
    outstr, _ = capsys.readouterr()
    assert out[0][0] == inp[0][0]
    assert out[0][1] == inp[1][0]
    assert out[0][2] == inp[2][0]
    assert out[0][3] == inp[3]
    assert out[0][4] == inp[4]
    assert_allclose(out[1], 0.027415567780804)
    assert_allclose(out[2], 0.018277045187203)
    assert out[3] == 1
    assert out[4] == 1
    assert out[5] == 1
    assert outstr[:42] == "   Receiver(s)     :  1 dipole(s)\n     > x"

    # Bipole, src, ninpz = 1, intpts = 5, verbose
    inp = [np.array([-50]), np.array([50]), np.array([50]), np.array([100]),
           np.array([0]), np.array([0])]
    out = utils.get_azm_dip(inp, 0, 1, 5, False, 300, 'src', 4)
    outstr, _ = capsys.readouterr()
    assert_allclose(out[0][0],
                    np.array([-45.309, -26.923, 0., 26.923, 45.309]))
    assert_allclose(out[0][1], np.array([52.346, 61.538, 75., 88.462, 97.654]))
    assert_allclose(out[0][2], np.array([0.,  0.,  0.,  0.,  0.]))
    assert_allclose(out[1], 0.463647609000806)
    assert out[2] == 0
    assert_allclose(out[3], np.array([0.118463442528094, 0.239314335249683,
                    0.284444444444445, 0.239314335249683, 0.118463442528094]))
    assert out[4] == 5
    assert_allclose(out[5], 33541.01966249684483)
    assert outstr[:47] == "   Source(s)       :  1 bipole(s)\n     > intpts"

    # Bipole, rec, ninpz = 2, intpts = 1, verbose
    inp = [np.array([-50, 0]), np.array([50, 0]), np.array([0, -50]),
           np.array([0, 50]), np.array([0, 100]), np.array([0, 100])]
    out = utils.get_azm_dip(inp, 0, 2, 1, False, 300, 'rec', 4)
    outstr, _ = capsys.readouterr()
    assert out[0][0] == 0
    assert out[0][1] == 0
    assert out[0][2] == 0
    assert out[1] == 0
    assert out[2] == 0
    assert out[3] == 1
    assert out[4] == 1
    assert out[5] == 100
    assert outstr[:47] == "   Receiver(s)     :  1 bipole(s)\n     > intpts"


def test_get_kwargs(capsys):
    kwargs1 = {'ft': 'sin', 'depth': []}
    ft, ht = utils.get_kwargs(['ft', 'ht'], ['dlf', 'dlf'], kwargs1)
    out, _ = capsys.readouterr()
    assert ft == 'sin'
    assert ht == 'dlf'
    assert "* WARNING :: Unused **kwargs: {'depth': []}" in out

    kwargs2 = {'depth': [], 'unknown': 1}
    with pytest.raises(TypeError, match='Unexpected '):
        utils.get_kwargs(['verb', ], [0, ], kwargs2)


def test_printstartfinish(capsys):
    t0 = utils.printstartfinish(0)
    assert isinstance(t0, float)
    out, _ = capsys.readouterr()
    assert out == ""

    t0 = utils.printstartfinish(3)
    assert isinstance(t0, float)
    out, _ = capsys.readouterr()
    assert ":: empymod START  ::  v" in out

    utils.printstartfinish(0, t0)
    out, _ = capsys.readouterr()
    assert out == ""

    utils.printstartfinish(3, t0)
    out, _ = capsys.readouterr()
    assert out[:27] == "\n:: empymod END; runtime = "

    utils.printstartfinish(3, t0, 13)
    out, _ = capsys.readouterr()
    assert out[-19:] == "13 kernel call(s)\n\n"


def test_conv_warning(capsys):
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
    assert '* WARNING :: Hankel-quadrature did not converge' in out

    # If converged, and verb>1, no output
    utils.conv_warning(True, ['', '', '', 51, ''], 'Hankel', 1)
    out, _ = capsys.readouterr()
    assert out == ""


def test_check_shape():
    # Ensure no Error is raised
    utils._check_shape(np.zeros((3, 4)), 'tvar', (3, 4))
    utils._check_shape(np.zeros((3, 4)), 'tvar', (3, 4), (2, ))
    utils._check_shape(np.zeros((3, 4)), 'tvar', (2,), (3, 4))
    # Ensure Error is raised
    with pytest.raises(ValueError, match='Parameter tvar has wrong shape'):
        utils._check_shape(np.zeros((3, 4)), 'tvar', (2,))
    with pytest.raises(ValueError, match='Parameter tvar has wrong shape'):
        utils._check_shape(np.zeros((3, 4)), 'tvar', (2,), (1, 4))


def test_check_var():
    # This is basically np.array(), with an optional call to _check_shape
    # above. Just three simple checks therefore; one without call, one with one
    # shape, and one with two shapes

    # Without shapes
    out = utils._check_var(np.pi, int, 3, 'tvar')
    assert out[0, 0, 0] == 3

    # One shape, but wrong
    with pytest.raises(ValueError, match='Parameter tvar has wrong shape'):
        out = utils._check_var(np.arange(3)*.5, float, 1, 'tvar', (1, 3))

    # Two shapes, second one is correct
    out = utils._check_var(np.arange(3)*.5, float, 1, 'tvar', (1, 3), (3, ))


def test_strvar():
    out = utils._strvar(np.arange(3)*np.pi)
    assert out == "0 3.14159 6.28319"

    out = utils._strvar(np.pi, '{:20.10e}')
    assert out == "    3.1415926536e+00"


def test_prnt_min_max_val(capsys):
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


def test_check_min(capsys):
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


def test_minimum():
    # Check default values
    d = utils.get_minimum()
    assert d['min_freq'] == 1e-20
    assert d['min_time'] == 1e-20
    assert d['min_off'] == 1e-3
    assert d['min_res'] == 1e-20
    assert d['min_angle'] == 1e-10

    # Set all default values to new values
    utils.set_minimum(1e-2, 1e-3, 1, 1e-4, 1e-5)

    # Check new values
    d = utils.get_minimum()
    assert d['min_freq'] == 1e-2
    assert d['min_time'] == 1e-3
    assert d['min_off'] == 1
    assert d['min_res'] == 1e-4
    assert d['min_angle'] == 1e-5


def test_report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    # Reporting is now done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    out1 = scooby.Report(
            core=['numpy', 'scipy', 'numba', 'empymod', 'libdlf'],
            optional=['IPython', 'matplotlib'],
            ncol=3)
    out2 = utils.Report()

    # Ensure they're the same; exclude time to avoid errors.
    assert out1.__repr__()[115:] == out2.__repr__()[115:]


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Not Linux.")
def test_import_time():
    # Relevant for the CLI: How long does it take to import?
    cmd = ["time", "-f", "%U", "python", "-c", "import empymod"]
    # Run it twice, just in case.
    subprocess.run(cmd)
    subprocess.run(cmd)
    # Capture it
    out = subprocess.run(cmd, capture_output=True)

    # Currently we check t < 1.2s.
    # => That should come down to t < 0.5s in the future!
    assert float(out.stderr.decode("utf-8")[:-1]) < 1.2


def test_all_dir():
    assert set(utils.__all__) == set(dir(utils))
