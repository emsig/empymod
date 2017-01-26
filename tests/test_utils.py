# utils. Status: 3/21
import pytest
import numpy as np
from numpy.testing import assert_allclose

from empymod import utils

# Notes:
# - Error and Warning print-statements are checked
# - Information print-statements are not checked


# 1. EMArray

# 2. check_time

# 3. check_model

# 4. check_frequency

# 5. check_hankel

# 6. check_opt

# 7. check_dipole

# 8. check_bipole

# 9. check_ab
def test_check_ab():
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

    # Check it raises a ValueError if a non-existing ab is provided.
    with pytest.raises(ValueError):
        utils.check_ab(77, 0)

    # We just check one other thing here, that it fails with a TypeError if a
    # list instead of one value is provided. Generally the try/except statement
    # with int() should take proper care of all the checking right in check_ab.
    with pytest.raises(TypeError):
        utils.check_ab([12, ], 0)


# 10. get_abs

# 11. get_geo_fact

# 12. get_azm_dip

# 13. get_off_ang

# 14. get_layer_nr

# 15. printstartfinish

# 16. conv_warning
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
    assert out[:35] == "* WARNING :: Hankel-QWE used all 51"

    # If converged, and verb>1, no output
    utils.conv_warning(True, ['', '', '', 51, ''], 'Hankel', 1)
    out, _ = capsys.readouterr()
    assert out == ""

# 17. _check_shape

# 18. _check_var

# 19. _strvar

# 20. _prnt_min_max_val


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
