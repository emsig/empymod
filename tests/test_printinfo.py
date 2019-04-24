import pip

# Optional imports
try:
    import IPython
except ImportError:
    IPython = False

from empymod.scripts import versions, Versions


def test_versions(capsys):

    # Check the default
    out1 = Versions().__repr__()

    # Check one of the standard packages
    assert 'numpy' in out1

    # Check the 'auto'-version, providing a package
    out1b = Versions(add_pckg=(pip, )).__repr__()

    # Check the provided package, with number
    assert pip.__version__ + ' : pip' in out1b

    # Check html-version, providing a package as a list
    out2 = Versions(add_pckg=[pip, ])._repr_html_()
    out2b = versions('HTML', add_pckg=pip)._repr_html_()  # Backwards test
    assert out2b == out2
    assert 'numpy' in out2
    assert 'td style=' in out2

    # Check row of provided package, with number
    teststr = "<td style='text-align: right; background-color: #ccc; "
    teststr += "border: 2px solid #fff;'>"
    teststr += pip.__version__
    teststr += "</td>\n    <td style='"
    teststr += "text-align: left; border: 2px solid #fff;'>pip</td>"
    assert teststr in out2
