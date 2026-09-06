import numpy as np
import pytest
from dataclasses import replace


from qkd.sources import (
    OpticalSource,
    DensityMatrixSource,
    PoissonSource,
    _validate_density_matrix,
)

from qkd.exceptions import ParameterValidationError
from qkd.datatypes import (
    OpticalSourceConfig,
    PulseTypeConfig,
    SourceStatisticsType,
    SourceErrorModel,
)

# -------------------------
# helpers
# -------------------------

class DummyMZM:
    def validate(self) -> None:
        return None

    def apply(self, mus, voltage_noise_std, rng):
        return mus




class DummyNoise:
    def validate(self) -> None:
        return None

    def total_voltage_std(self) -> float:
        return 0.0




def build_config():
    pulses = (
        PulseTypeConfig(name="signal", mean_photon_number=0.5, probability=0.7),
        PulseTypeConfig(name="decoy", mean_photon_number=0.1, probability=0.3),
    )

    return OpticalSourceConfig(
        pulse_configs=pulses,
        source_rate=1e9,
        is_bidirectional=False,
        statistics_type=SourceStatisticsType.POISSON,
        intensity_jitter=0.0,
        error_model=SourceErrorModel.RANDOM_GAUSSIAN,
        modulation_index=0.0,
        N_channels=1,
        use_small_angle_approximation=True,
        use_linear_modulation_approximation=False,
        ideal_emission_probability=1.0,
        adversarial_block_size=1,
        mzm=DummyMZM(),
        electrical_noise=DummyNoise(),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def config():
    return build_config()


# -------------------------
# density matrix validation
# -------------------------

def test_validate_density_matrix_valid():
    rho = np.array([[0.7, 0.0], [0.0, 0.3]])
    _validate_density_matrix(rho)


def test_validate_density_matrix_not_hermitian():
    rho = np.array([[0.5, 1.0], [0.0, 0.5]])
    with pytest.raises(ParameterValidationError):
        _validate_density_matrix(rho)


def test_validate_density_matrix_trace_not_one():
    rho = np.array([[0.5, 0.0], [0.0, 0.6]])
    with pytest.raises(ParameterValidationError):
        _validate_density_matrix(rho)


def test_validate_density_matrix_negative_eigenvalue():
    rho = np.array([[1.2, 0.0], [0.0, -0.2]])
    with pytest.raises(ParameterValidationError):
        _validate_density_matrix(rho)


# -------------------------
# OpticalSource basic tests
# -------------------------

def test_optical_source_init(config):
    src = OpticalSource.from_config(config)

    assert len(src.pulse_configs) == 2
    assert src.statistics_type == SourceStatisticsType.POISSON
    assert np.allclose(src._base_mus_cache, [0.5, 0.1])
    assert np.allclose(src._probabilities_cache, [0.7, 0.3])


def test_get_pulse_config_by_name(config):
    src = OpticalSource.from_config(config)
    pc = src.get_pulse_config_by_name("signal")

    assert pc.mean_photon_number == 0.5

    with pytest.raises(ValueError):
        src.get_pulse_config_by_name("unknown")


# -------------------------
# RNG validation
# -------------------------

def test_validate_rng_seed_valid():
    OpticalSource.validate_rng_seed(123)


def test_validate_rng_seed_invalid_type():
    with pytest.raises(ParameterValidationError):
        OpticalSource.validate_rng_seed("bad")


def test_validate_rng_seed_invalid_range():
    with pytest.raises(ParameterValidationError):
        OpticalSource.validate_rng_seed(-1)


# -------------------------
# block size calculation
# -------------------------

def test_calculate_minimum_block_size():
    size = OpticalSource.calculate_minimum_block_size(0.5)
    assert size > 0


def test_calculate_minimum_block_size_invalid():
    with pytest.raises(ParameterValidationError):
        OpticalSource.calculate_minimum_block_size(-1)


# -------------------------
# photon generation
# -------------------------

def test_generate_photons_scalar(config, rng):
    src = OpticalSource.from_config(config)
    photons = src.generate_photons(0, rng)

    assert isinstance(photons, (int, np.integer))
    assert photons >= 0


def test_generate_photons_vector(config, rng):
    src = OpticalSource.from_config(config)
    photons = src.generate_photons(np.array([0, 1, 0, 1]), rng)

    assert photons.shape == (4,)
    assert np.all(photons >= 0)


def test_generate_photons_random_choice(config, rng):
    src = OpticalSource.from_config(config)
    photons = src.generate_photons(None, rng, num_samples=10)

    assert len(photons) == 10


# -------------------------
# ideal emission probability
# -------------------------

def test_nonideal_emission(config, rng):
    config = replace(build_config(), ideal_emission_probability=0.0)
    src = OpticalSource.from_config(config)

    photons = src.generate_photons(np.array([0]*100), rng)

    assert np.all(photons >= 0)


# -------------------------
# intensity jitter
# -------------------------

def test_intensity_jitter_gaussian(config, rng):
    config = replace(
        build_config(),
        intensity_jitter=0.2,
        error_model=SourceErrorModel.RANDOM_GAUSSIAN,
    )


    src = OpticalSource.from_config(config)

    mus = np.array([0.5, 0.5, 0.5])
    jittered = src._apply_intensity_jitter(mus, rng)

    assert jittered.shape == mus.shape
    assert np.all(jittered >= 0)


# -------------------------
# DensityMatrixSource
# -------------------------

def test_density_matrix_source_sampling(config, rng):
    rho = np.array([
        [0.7, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.1],
    ])

    dm = {"signal": rho}

    src = DensityMatrixSource.from_config(config, density_matrices=dm)

    photons = src.generate_photons_dm(np.array([0,0,0]), rng)

    assert photons.shape == (3,)
    assert np.all(photons >= 0)


def test_density_matrix_source_fallback(config, rng):
    src = DensityMatrixSource.from_config(config, density_matrices=None)

    photons = src.generate_photons_dm(0, rng)

    assert photons >= 0


# -------------------------
# PoissonSource
# -------------------------

def test_poisson_source_valid(config):
    src = PoissonSource.from_config(config)
    assert isinstance(src, PoissonSource)


def test_poisson_source_invalid_statistics(config):
    config = replace(build_config(), statistics_type=SourceStatisticsType.THERMAL)

    with pytest.raises(ParameterValidationError):
        PoissonSource.from_config(config)


# -------------------------
# serialization
# -------------------------

def test_to_config_dict(config):
    src = OpticalSource.from_config(config)
    d = src.to_config_dict()

    assert "pulse_configs" in d

