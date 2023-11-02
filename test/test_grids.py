from speed.grids import GLOBAL

import numpy as np

def test_global_grid():
    """
    Esure that speed.grids.GLOBAL satisfies:
        - lons in [-180, 180]
        - lats in [-90, 90]
        - Both lons and lats are uniformly spaced.
        - Sizes of lons and lats match grid of merged IR data.
    """
    lons = GLOBAL.lons
    assert ((lons >= -180) * (lons <= 180)).all()
    d_lons = np.diff(lons)
    assert np.isclose(d_lons, d_lons[0]).all()
    assert lons.size == 9896

    lats = GLOBAL.lats
    assert ((lats >= -90) * (lats <= 90)).all()
    d_lats = np.diff(lats)
    assert np.isclose(d_lats, d_lats[0]).all()
    assert lats.size == 3298
