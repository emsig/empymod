import os
import pytest
import numpy as np
from os.path import join
from numpy.testing import assert_allclose
from contextlib import ContextDecorator

import empymod
from empymod.__main__ import run


class disable_numba(ContextDecorator):
    """Context decorator to disable-enable JIT and remove log file."""
    def __enter__(self):
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        return self

    def __exit__(self, *exc):
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        return False


@disable_numba()
@pytest.mark.script_launch_mode('subprocess')
def test_main(script_runner):

    # help
    for inp in ['--help', '-h']:
        ret = script_runner.run('empymod', inp)
        assert ret.success
        assert "3D electromagnetic modeller for 1D VTI media" in ret.stdout

    # info
    ret = script_runner.run('empymod')
    assert ret.success
    assert "3D electromagnetic modeller for 1D VTI media." in ret.stdout
    assert "empymod v" in ret.stdout

    # report
    ret = script_runner.run('empymod', '--report')
    assert ret.success
    # Exclude time to avoid errors.
    # Exclude empymod-version (after 300), because if run locally without
    # having empymod installed it will be "unknown" for the __main__ one.
    assert empymod.utils.Report().__repr__()[115:300] in ret.stdout

    # version        -- VIA empymod/__main__.py by calling the folder empymod.
    ret = script_runner.run('python', 'empymod', '--version')
    assert ret.success
    assert "empymod v" in ret.stdout

    # Wrong function -- VIA empymod/__main__.py by calling the file.
    ret = script_runner.run(
            'python', join('empymod', '__main__.py'), 'wrong')
    assert not ret.success
    assert "error: argument routine: invalid choice: 'wrong'" in ret.stderr

    # try to run
    ret = script_runner.run(
            'empymod', 'bipole', 'test.json', 'output.txt')
    assert not ret.success
    assert "No such file or directory" in ret.stderr


class TestRun:

    def test_bipole_txt(self, tmpdir):

        inp = {
            'src': [0, 0, 0, 0, 0],
            'rec': [100, 50, 10, 0, 0],
            'depth': [-20, 20],
            'res': [2e14, 1, 100],
            'freqtime': 0.01,
            'htarg': {'dlf': 'wer_201_2018', 'pts_per_dec': -1},
            'msrc': True,
            'mrec': True,
            'signal': None,
            'strength': np.pi,
            'srcpts': 5,
        }
        empymod.io.save_input(join(tmpdir, 't.json'), inp)

        args_dict = {
            'routine': 'bipole',
            'input': join(tmpdir, 't.json'),
            'output': join(tmpdir, 'out.txt')
        }
        run(args_dict)
        out = empymod.io.load_data(join(tmpdir, 'out.txt'))
        assert_allclose(out, empymod.bipole(**inp))

    def test_dipole_stdout(self, tmpdir, capsys):

        inp = {
            'src': [0, 0, 0],
            'rec': [100, 50, 10],
            'depth': [-20, 20],
            'res': [2e14, 1, 100],
            'ab': 12,
            'freqtime': 10,
            'verb': 1,
        }
        empymod.io.save_input(join(tmpdir, 't.json'), inp)

        args_dict = {
            'routine': 'dipole',
            'input': join(tmpdir, 't.json'),
            'output': None
        }
        _, _ = capsys.readouterr()
        run(args_dict)
        out, _ = capsys.readouterr()
        out = complex(out.strip().strip("[").strip("]"))
        assert_allclose(out, empymod.dipole(**inp))

    def test_loop_txt(self, tmpdir):

        inp = {
            'src': [0, 0, 0, 0, 0],
            'rec': [100, 50, 10, 0, 0],
            'depth': [-20, 20],
            'res': [2e14, 1, 100],
            'freqtime': 0.01,
        }
        empymod.io.save_input(join(tmpdir, 't.json'), inp)

        args_dict = {
            'routine': 'loop',
            'input': join(tmpdir, 't.json'),
            'output': join(tmpdir, 'out.txt')
        }
        run(args_dict)
        out = empymod.io.load_data(join(tmpdir, 'out.txt'))
        assert_allclose(out, empymod.loop(**inp))

    def test_analytical_json(self, tmpdir):

        inp = {
            'src': [0, 0, 0],
            'rec': [100, 50, 10],
            'res': np.pi,
            'freqtime': np.pi,
        }
        empymod.io.save_input(join(tmpdir, 't.json'), inp)

        args_dict = {
            'routine': 'analytical',
            'input': join(tmpdir, 't.json'),
            'output': join(tmpdir, 'out.json')
        }
        run(args_dict)
        out = empymod.io.load_data(join(tmpdir, 'out.json'))
        assert_allclose(out, empymod.analytical(**inp))

    def test_failure(self, tmpdir):

        args_dict = {
            'routine': 'bipole',
            'input': join(tmpdir, 't.json'),
            'output': join(tmpdir, 'out.json')
        }

        with pytest.raises(FileNotFoundError, match="t.json'"):
            run(args_dict)
