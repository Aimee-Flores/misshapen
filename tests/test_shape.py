# -*- coding: utf-8 -*-
"""
Tests for shape.py
"""

import pytest
import numpy as np
from misshapen import shape


def test_rdratio():
    """
    Tests
    -----
    1. Pseudoinput
    2. Error: Unequal number of peaks and troughs
    """
    
    # 1. Pseudoinput
    Ps = [1, 12, 17]
    Ts = [5, 15, 25]
    riset, decayt, rdr = shape.rdratio(Ps, Ts)
    assert np.allclose(riset, np.allclose(riset,np.array([7,2])), atol=10 ** -5)
    assert np.allclose(decayt, np.allclose(riset,np.array([4,3])), atol=10 ** -5)
    assert np.allclose(rdr, np.allclose(riset,np.array([7/4.,2/3.])), atol=10 ** -5)

    # Error: uneuqal peaks and troughs
    Ps = [1, 11, 21, 32]
    Ts = [5, 15, 25]
    with pytest.raises(ValueError) as excinfo:
        shape.rdratio(Ps, Ts)
    assert 'Length of peaks and troughs' in str(excinfo.value)