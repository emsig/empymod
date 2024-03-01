import pytest
import numpy as np
from numpy.testing import assert_allclose

import empymod
from empymod import io, filters


class TestSaveLoadInput:

    def test_basic(self, tmpdir):

        inp = {
            "src": [[0, 0], [0, 1000], -250, 45, 0],
            "rec": [[1000, 2000, 3000], [0, 0, 0], -300, 0, 0],
            "depth": [0, 300, 1000, 1200],
            "res": [2e14, 0.3, 1, 50, 1],
            "freqtime": [0.1, 1, 10, 100],
            "signal": None,
            "msrc": True,
            "htarg": {"pts_per_dec": -1},
        }

        io.save_input(tmpdir+'/test.json', data=inp)
        out = io.load_input(tmpdir+'/test.json')

        # Won't work with, e.g., np-arrays
        # (the comparison; {save;load}_input does work).
        assert inp == out

        # Dummy check by comparing the produced result from the two inputs.
        assert_allclose(empymod.bipole(**inp), empymod.bipole(**out), 0, 0)

        # Test filter instance
        inp["htarg"] = {"dlf": filters.Hankel().wer_201_2018}
        io.save_input(tmpdir+'/test.json', data=inp)
        out = io.load_input(tmpdir+'/test.json')
        assert out["htarg"]["dlf"] == filters.Hankel().wer_201_2018.name
        assert_allclose(empymod.bipole(**inp), empymod.bipole(**out), 0, 0)

    def test_errors(self, tmpdir):

        with pytest.raises(ValueError, match="Unknown extension '.abc'"):
            io.save_input(tmpdir+'/test.abc', data=1)

        with pytest.raises(ValueError, match="Unknown extension '.abc'"):
            io.load_input(tmpdir+'/test.abc')


class TestSaveLoadData:

    inp = {
        'src': ([0, 111, 1111], [0, 0, 0], 250),
        'rec': [np.arange(1, 8)*1000, np.zeros(7), 300],
        'depth': [0, 300],
        'res': [2e14, 0.3, 1],
        'htarg': {'pts_per_dec': -1},
        'verb': 1,
    }

    def test_basic(self, tmpdir):

        # Compute
        orig = empymod.dipole(**self.inp, freqtime=[0.1, 1, 10, 100])

        # Save
        io.save_data(tmpdir+'test.txt', orig, info='Additional info')
        io.save_data(tmpdir+'test.json', orig, info='Additional info')

        # Load
        orig_txt = io.load_data(tmpdir+'test.txt')
        orig_json = io.load_data(tmpdir+'test.json')

        # Compare numbers
        assert_allclose(orig, orig_txt)
        assert_allclose(orig, orig_json)

        # Ensure some header things

        for ending in ['txt', 'json']:
            with open(tmpdir+'test.'+ending, 'r') as f:
                text = f.read()

            assert 'date' in text
            assert 'empymod v' in text
            assert 'shape' in text
            assert '(4, 7, 3)' in text
            assert str(orig.dtype) in text
            assert 'Additional info' in text

    def test_text(self, tmpdir):

        # Compute
        orig = empymod.dipole(**self.inp, freqtime=[0.1, 1, 10, 100])

        # Save
        io.save_data(tmpdir+'test.txt', orig)
        io.save_data(tmpdir+'test.json', orig)

    def test_errors(self, tmpdir):

        with pytest.raises(ValueError, match="must be 3D"):
            io.save_data(tmpdir+'/test.json', data=np.ones((1, 1)))

        with pytest.raises(ValueError, match="Unknown extension '.abc'"):
            io.save_data(tmpdir+'/test.abc', data=np.ones((1, 1, 1)))

        with pytest.raises(ValueError, match="Unknown extension '.abc'"):
            io.load_data(tmpdir+'/test.abc')


def test_ComplexNumPyEncoder():

    test = io._ComplexNumPyEncoder()

    # NumPy types
    assert type(test.default(np.int_(1))) is int
    assert type(test.default(np.float_(1))) is float
    assert type(test.default(np.bool_(1))) is bool
    assert type(test.default(np.array([[1., 1.], [1., 1.]]))) is list

    # Complex values
    cplx = test.default(np.array([[[1+1j]]]))
    assert type(cplx) is list
    assert type(cplx[0][0][0][0]) is float

    # Error
    with pytest.raises(TypeError, match="Object of type module"):
        test.default(io)


def test_all_dir():
    assert set(io.__all__) == set(dir(io))
