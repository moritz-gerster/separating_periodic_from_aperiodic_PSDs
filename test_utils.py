import numpy as np
from utils import elec_phys_signal


# Test simulation of electrophysiological signals
def test_elec_phys_signal():

    # test output
    output = elec_phys_signal(1)
    assert isinstance(output, tuple)
    assert isinstance(output[0], np.ndarray)
    assert isinstance(output[1], np.ndarray)

    # test impact of 1/f exponent
    assert not np.allclose(elec_phys_signal(1)[0],
                           elec_phys_signal(2)[0])

    # test impact of periodic_params
    params1 = dict(exponent=1, periodic_params=[(1, 1, 1), (2, 2, 2)])
    params2 = dict(exponent=1, periodic_params=[(3, 3, 3), (2, 2, 2)])
    aperiodic_signal1, full_signal1 = elec_phys_signal(**params1)
    aperiodic_signal2, full_signal2 = elec_phys_signal(**params2)
    assert not np.allclose(full_signal1, full_signal2)
    assert np.allclose(aperiodic_signal1, aperiodic_signal2)

    # test impact of noise level
    assert not np.allclose(elec_phys_signal(1, nlv=1)[0],
                           elec_phys_signal(1, nlv=2)[0])

    # test impact of highpass
    assert not np.allclose(elec_phys_signal(1, highpass=1)[0],
                           elec_phys_signal(1, highpass=0)[0])

    # test duration of signal
    assert elec_phys_signal(1, sample_rate=24, duration=1)[0].shape[0] == 24-2
    assert elec_phys_signal(1, sample_rate=1, duration=24)[0].shape[0] == 24-2

    # test impact of seed
    assert not np.allclose(elec_phys_signal(1, seed=0)[0],
                           elec_phys_signal(1, seed=1)[0])