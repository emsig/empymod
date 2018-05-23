import pytest

# Optional imports
try:
    import IPython
except ImportError:
    IPython = False

from empymod.scripts import versions, printinfo


def test_versions(capsys):

    # Check the 'auto'-version, providing a package
    versions(add_pckg=pytest)
    out1, _ = capsys.readouterr()

    # Check one of the standard packages
    assert 'numpy' in out1

    # Check the provided package, with number
    assert pytest.__version__ + ' : pytest' in out1

    # Check the 'text'-version, providing a package as tuple
    versions('print', add_pckg=(pytest, ))
    out2, _ = capsys.readouterr()

    # They have to be the same, except time (run at slightly different times)
    assert out1[:-75] == out2[:-75]

    # Check the 'Pretty'/'plain'-version, providing a package as list
    out3 = versions('plain', add_pckg=[pytest, ])
    out3b = printinfo.versions_text(add_pckg=[pytest, ])
    out3c = versions('Pretty', add_pckg=[pytest, ])

    # They have to be the same, except time (run at slightly different times)
    assert out3[:-75] == out3b[:-75]
    if IPython:
        assert out3[:-75] == out3c.data[:-75]
    else:
        assert out3c is None

    # Check one of the standard packages
    assert 'numpy' in out3

    # Check the provided package, with number
    assert pytest.__version__ + ' : pytest' in out3

    # Check 'HTML'/'html'-version, providing a package as a list
    out4 = versions('html', add_pckg=[pytest])
    out4b = printinfo.versions_html(add_pckg=[pytest])
    out4c = versions('HTML', add_pckg=[pytest])

    assert 'numpy' in out4
    assert 'td style=' in out4

    # They have to be the same, except time (run at slightly different times)
    assert out4[:-50] == out4b[:-50]
    if IPython:
        assert out4[:-50] == out4c.data[:-50]
    else:
        assert out4c is None

    # Check row of provided package, with number
    teststr = "<td style='background-color: #ccc; border: 2px solid #fff;'>"
    teststr += pytest.__version__
    teststr += "</td>\n    <td style='"
    teststr += "border: 2px solid #fff; text-align: left;'>pytest</td>"
    assert teststr in out4
