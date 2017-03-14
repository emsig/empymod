import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from empymod import transform, filters, kernel, utils

# No input checks are carried out in transform, by design. Input checks are
# carried out in model/utils, not in the core functions kernel/transform.
# Rubbish in, rubbish out. So we also do not check these functions for wrong
# inputs.

# Load required data
# Data generated with create_transform.py [28/01/2017]
DATA = np.load(join(dirname(__file__), 'data_transform.npz'))


@pytest.mark.parametrize("htype", ['fht', 'hqwe', 'hquad'])
def test_fht(htype):                              # 1. fht / 2. hqwe / 3. hquad
    # Compare wavenumber-domain calculation / FHT with analytical
    # frequency-domain fullspace solution
    calc = getattr(transform, htype)
    model = utils.check_model([], 10, 2, 2, 5, 1, 10, 0)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model
    frequency = utils.check_frequency(1, res, aniso, epermH, epermV, mpermH,
                                      mpermV, 0)
    freq, etaH, etaV, zetaH, zetaV = frequency
    src = [0, 0, 0]
    src, nsrc = utils.check_dipole(src, 'src', 0)
    ab, msrc, mrec = utils.check_ab(11, 0)
    ht, htarg = utils.check_hankel(htype, None, 0)
    options = utils.check_opt(None, None, htype, None, 0)
    use_spline, use_ne_eval, loop_freq, loop_off = options
    xdirect = False  # Important, as we want to compare wavenumber-frequency!
    rec = [np.arange(1, 11)*500, np.zeros(10), 300]
    rec, nrec = utils.check_dipole(rec, 'rec', 0)
    off, angle = utils.get_off_ang(src, rec, nsrc, nrec, 0)
    lsrc, zsrc = utils.get_layer_nr(src, depth)
    lrec, zrec = utils.get_layer_nr(rec, depth)

    # # # 0. No Spline # # #
    if htype != 'hquad':  # hquad is always using spline
        # Wavenumber solution plus transform
        wvnr0, _, conv = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab,
                              etaH, etaV, zetaH, zetaV, xdirect, htarg,
                              use_spline, False, msrc, mrec)
        # Analytical frequency-domain solution
        freq0 = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH,
                                 zetaV, ab, msrc, mrec)
        # Compare
        assert_allclose(conv, True)
        assert_allclose(np.squeeze(wvnr0), np.squeeze(freq0))

    # # # 1. Spline; One angle # # #
    options = utils.check_opt('spline', None, htype, None, 0)
    use_spline, use_ne_eval, loop_freq, loop_off = options
    use_spline, use_ne_eval, loop_freq, loop_off = options
    # Wavenumber solution plus transform
    wvnr1, _, conv = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab, etaH,
                          etaV, zetaH, zetaV, xdirect, htarg, use_spline,
                          False, msrc, mrec)
    # Analytical frequency-domain solution
    freq1 = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV,
                             ab, msrc, mrec)
    # Compare
    assert_allclose(conv, True)
    assert_allclose(np.squeeze(wvnr1), np.squeeze(freq1), rtol=1e-4)

    # # # 2. Spline; Multi angle # # #
    rec = [np.arange(1, 11)*500, np.arange(-5, 5)*200, 300]
    rec, nrec = utils.check_dipole(rec, 'rec', 0)
    off, angle = utils.get_off_ang(src, rec, nsrc, nrec, 0)
    if htype == 'hqwe':
        ht, htarg = utils.check_hankel('qwe', ['', '', '', 200], 0)
    # Analytical frequency-domain solution
    wvnr2, _, conv = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab, etaH,
                          etaV, zetaH, zetaV, xdirect, htarg, use_spline,
                          False, msrc, mrec)
    # Analytical frequency-domain solution
    freq2 = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV,
                             ab, msrc, mrec)
    # Compare
    assert_allclose(conv, True)
    assert_allclose(np.squeeze(wvnr2), np.squeeze(freq2), rtol=1e-4)

    # # # 3. Spline; pts_per_dec # # #
    if htype == 'fht':
        ht, htarg = utils.check_hankel('fht', ['key_201_2012', 20], 0)
    elif htype == 'hqwe':
        ht, htarg = utils.check_hankel('qwe', ['', '', '', '', 100], 0)
    if htype != 'hquad':  # hquad is always pts_per_dec
        # Analytical frequency-domain solution
        wvnr3, _, conv = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab,
                              etaH, etaV, zetaH, zetaV, xdirect, htarg,
                              use_spline, False, msrc, mrec)
        # Analytical frequency-domain solution
        freq3 = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH,
                                 zetaV, ab, msrc, mrec)
        # Compare
        assert_allclose(conv, True)
        assert_allclose(np.squeeze(wvnr3), np.squeeze(freq3), rtol=1e-4)

    # # # 4. Spline; Only one offset # # #
    rec = [5000, 0, 300]
    rec, nrec = utils.check_dipole(rec, 'rec', 0)
    off, angle = utils.get_off_ang(src, rec, nsrc, nrec, 0)
    if htype == 'hqwe':
        ht, htarg = utils.check_hankel('qwe', ['', '', '', 200, ''], 0)
    elif htype == 'hquad':
        ht, htarg = utils.check_hankel('quad', None, 0)
    # Analytical frequency-domain solution
    wvnr4, _, conv = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab, etaH,
                          etaV, zetaH, zetaV, xdirect, htarg, use_spline,
                          False, msrc, mrec)
    # Analytical frequency-domain solution
    freq4 = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV,
                             ab, msrc, mrec)
    # Compare
    assert_allclose(conv, True)
    assert_allclose(np.squeeze(wvnr4), np.squeeze(freq4), rtol=1e-4)


@pytest.mark.parametrize("ftype", ['fft', 'fqwe', 'fftlog'])
def test_fft(ftype):                             # 3. fft / 4. fqwe / 5. fftlog
    # Check FFT-method with the analytical functions for a halfspace.
    t = DATA['t'][()]
    for i in [0, 1]:
        fl = DATA[ftype+str(i)][()]
        res = DATA['tEM'+str(i)][()]
        finp = fl['fEM']
        if i == 1:
            finp /= 2j*np.pi*fl['f']
        tEM, _ = getattr(transform, ftype)(finp, t, fl['f'], fl['ftarg'])
        assert_allclose(tEM*2/np.pi, res, rtol=1e-3)


def test_qwe():                                                        # 6. qwe
    # QWE is integral of hqwe and fqwe, and therefore tested a lot through
    # those. Here we just ensure status quo. And if a problem arises in hqwe or
    # fqwe, it would make it obvious if the problem arises from qwe or not.

    # Fourier
    dat = DATA['fqwe0'][()]
    tres = DATA['tEM0'][()]
    ftarg = dat['ftarg']
    tEM, _, _ = transform.qwe(ftarg[0], ftarg[1], ftarg[3], dat['sEM'],
                              dat['intervals'])
    assert_allclose(np.squeeze(-tEM*2/np.pi), tres, rtol=1e-3)

    # Hankel
    dat = DATA['hqwe'][()]

    # With spline
    fEM, _, _ = transform.qwe(dat['rtol'], dat['atol'], dat['maxint'],
                              dat['getkernel'], dat['intervals'], None, None,
                              None)
    assert_allclose(np.squeeze(fEM), dat['freqres'], rtol=1e-5)

    # Without spline
    # Define getkernel here, no straightforward way to pickle it...
    def getkernel(i, inplambd, inpoff, inpfang):
        iB = i*dat['nquad'] + np.arange(dat['nquad'])
        PJ = kernel.wavenumber(lambd=np.atleast_2d(inplambd)[:, iB],
                               **dat['nsinp'])
        fEM = inpfang*np.dot(PJ[1][0, :], dat['BJ1'][iB])
        if dat['ab'] in [11, 12, 21, 22, 14, 24, 15, 25]:
            fEM /= np.atleast_1d(inpoff)
        fEM += inpfang*np.dot(PJ[2][0, :], dat['BJ0'][iB])
        fEM += np.dot(PJ[0][0, :], dat['BJ0'][iB])
        return fEM
    fEM, _, _ = transform.qwe(dat['rtol'], dat['atol'], dat['maxint'],
                              getkernel, dat['intervals'],
                              dat['lambd'], dat['off'], dat['factAng'])
    assert_allclose(np.squeeze(fEM), dat['freqres'], rtol=1e-5)


def test_get_spline_values():                            # 7. get_spline_values
    # Check one example
    filt = filters.key_81_CosSin_2009()
    out, new_inp = transform.get_spline_values(filt, np.arange(1, 6), None)
    # Expected values
    oout = np.array([[6.70925256e-05, 8.19469958e-05, 1.00090287e-04,
                      1.22250552e-04, 1.49317162e-04, 1.82376393e-04,
                      2.22755030e-04, 2.72073608e-04, 3.32311455e-04,
                      4.05886127e-04, 4.95750435e-04, 6.05510949e-04,
                      7.39572743e-04, 9.03316189e-04, 1.10331288e-03,
                      1.34758940e-03, 1.64594941e-03, 2.01036715e-03,
                      2.45546798e-03, 2.99911536e-03, 3.66312778e-03,
                      4.47415437e-03, 5.46474449e-03, 6.67465399e-03,
                      8.15244080e-03, 9.95741367e-03, 1.21620125e-02,
                      1.48547156e-02, 1.81435907e-02, 2.21606317e-02,
                      2.70670566e-02, 3.30597776e-02, 4.03793036e-02,
                      4.93193928e-02, 6.02388424e-02, 7.35758882e-02,
                      8.98657928e-02, 1.09762327e-01, 1.34064009e-01,
                      1.63746151e-01, 2.00000000e-01, 2.44280552e-01,
                      2.98364940e-01, 3.64423760e-01, 4.45108186e-01,
                      5.43656366e-01, 6.64023385e-01, 8.11039993e-01,
                      9.90606485e-01, 1.20992949e+00, 1.47781122e+00,
                      1.80500270e+00, 2.20463528e+00, 2.69274761e+00,
                      3.28892935e+00, 4.01710738e+00, 4.90650604e+00,
                      5.99282001e+00, 7.31964689e+00, 8.94023690e+00,
                      1.09196300e+01, 1.33372662e+01, 1.62901737e+01,
                      1.98968631e+01, 2.43020835e+01, 2.96826318e+01,
                      3.62544484e+01, 4.42812832e+01, 5.40852815e+01,
                      6.60599120e+01, 8.06857587e+01, 9.85498082e+01,
                      1.20369008e+02, 1.47019038e+02, 1.79569458e+02,
                      2.19326632e+02, 2.67886153e+02, 3.27196886e+02,
                      3.99639179e+02, 4.88120396e+02, 5.96191597e+02,
                      7.28190061e+02, 8.89413350e+02, 1.08633192e+03,
                      1.32684880e+03, 1.62061679e+03, 1.97942581e+03,
                      2.41767615e+03, 2.95295631e+03, 3.60674899e+03]])
    onew_inp = np.array([5., 4.09365377, 3.35160023, 2.74405818, 2.24664482,
                         1.83939721, 1.50597106, 1.23298482, 1.00948259,
                         0.82649444])
    # Comparison
    assert_allclose(out, oout)
    assert_allclose(new_inp, onew_inp)

    # Ensure out has dimension len(filter.base)+3, if inp is smaller than that
    filt = filters.anderson_801_1982()
    out, _ = transform.get_spline_values(filt, np.array([1, 1.1]), None)
    assert_allclose(out.size, 804)


def test_fhti():                                                      # 8. fhti
    # Check one example
    freq, tcalc, dlnr, kr, rk = transform.fhti(-1, 2, 60, 0)
    # Expected values
    ofreq = np.array([0.01685855, 0.0189156, 0.02122365, 0.02381333, 0.026719,
                      0.02997921, 0.03363722, 0.03774158, 0.04234675,
                      0.04751384, 0.05331141, 0.05981638, 0.06711508,
                      0.07530436, 0.08449288, 0.09480257, 0.10637024,
                      0.11934937, 0.1339122, 0.15025196, 0.16858547, 0.189156,
                      0.21223653, 0.2381333, 0.26718996, 0.29979206,
                      0.33637223, 0.37741585, 0.42346755, 0.4751384,
                      0.53311405, 0.59816381, 0.67115083, 0.75304362,
                      0.84492884, 0.94802575, 1.06370238, 1.1934937,
                      1.33912196, 1.50251955, 1.68585466, 1.89156004,
                      2.12236528, 2.38133301, 2.67189958, 2.99792064,
                      3.36372228, 3.77415847, 4.23467545, 4.75138401,
                      5.33114054, 5.98163807, 6.7115083, 7.53043617,
                      8.44928835, 9.48025746, 10.63702382, 11.93493702,
                      13.39121959, 15.0251955])
    otcalc = np.array([0.01094445, 0.01227988, 0.01377825, 0.01545945,
                       0.01734579, 0.01946229, 0.02183705, 0.02450157,
                       0.02749122, 0.03084566, 0.03460939, 0.03883238,
                       0.04357065, 0.04888707, 0.05485219, 0.06154517,
                       0.06905482, 0.07748078, 0.08693487, 0.09754253,
                       0.10944452, 0.12279877, 0.13778248, 0.15459449,
                       0.17345787, 0.19462293, 0.21837052, 0.24501575,
                       0.27491219, 0.30845655, 0.34609395, 0.38832379,
                       0.43570646, 0.48887069, 0.54852194, 0.61545174,
                       0.69054821, 0.77480783, 0.86934869, 0.97542527,
                       1.09444515, 1.22798766, 1.37782481, 1.54594487,
                       1.73457867, 1.94622928, 2.18370517, 2.4501575,
                       2.74912193, 3.08456554, 3.46093945, 3.88323794,
                       4.35706463, 4.88870692, 5.48521938, 6.15451737,
                       6.90548207, 7.74807832, 8.69348686, 9.75425268])
    odlnr = 0.11512925464970231
    okr = 1.0332228492019395,
    ork = 15.202880269326004
    # Comparison
    assert_allclose(freq, ofreq, atol=1e-7)
    assert_allclose(tcalc, otcalc, atol=1e-7)
    assert_allclose(dlnr, odlnr, atol=1e-7)
    assert_allclose(kr, okr, atol=1e-7)
    assert_allclose(rk, ork, atol=1e-7)
