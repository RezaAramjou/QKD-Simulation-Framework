
# === AUTO-GENERATED SCM BACKWARD COMPATIBILITY SHIM ===
import numpy as np
from .sources import OpticalSource, DensityMatrixSource, PulseEnsembleConfig
from .channel import FiberChannel
from .detectors import SinglePhotonDetector
from .datatypes import SourceStatisticsType, SourceErrorModel

def _copy_attrs(target, source):
    """Safely copy attributes between objects, fully supporting __slots__."""
    for cls in type(source).__mro__:
        if hasattr(cls, '__slots__'):
            for slot in cls.__slots__:
                if hasattr(source, slot):
                    try: setattr(target, slot, getattr(source, slot))
                    except AttributeError: pass
    if hasattr(source, '__dict__'):
        if hasattr(target, '__dict__'):
            target.__dict__.update(source.__dict__)
        else:
            for k, v in source.__dict__.items():
                try: setattr(target, k, v)
                except AttributeError: pass

# 1. Patch OpticalSource
_orig_opt_init = OpticalSource.__init__
def _patch_opt(self, *args, pulse_configs=None, source_rate=1e9, pulse_period_ns=None, statistics_type=SourceStatisticsType.POISSON, error_model=SourceErrorModel.RANDOM_GAUSSIAN, **kwargs):
    if pulse_period_ns is not None and source_rate == 1e9: source_rate = 1e9 / float(pulse_period_ns)
    if pulse_configs is not None:
        ensemble = PulseEnsembleConfig(pulses=list(pulse_configs))
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['intensity_jitter', 'modulation_index', 'N_channels', 'use_small_angle_approximation', 'use_linear_modulation_approximation', 'ideal_emission_probability', 'adversarial_block_size', 'is_bidirectional']}
        new_src = OpticalSource.from_pulse_ensemble(pulse_ensemble=ensemble, source_rate=source_rate, statistics_type=statistics_type, error_model=error_model, **valid_kwargs)
        _copy_attrs(self, new_src)
    else:
        _orig_opt_init(self, *args, **kwargs)
OpticalSource.__init__ = _patch_opt

# 2. Patch DensityMatrixSource
_orig_dm_init = DensityMatrixSource.__init__
def _patch_dm(self, *args, pulse_configs=None, source_rate=1e9, pulse_period_ns=None, statistics_type=SourceStatisticsType.POISSON, **kwargs):
    if pulse_period_ns is not None and source_rate == 1e9: source_rate = 1e9 / float(pulse_period_ns)
    if pulse_configs is not None:
        ensemble = PulseEnsembleConfig(pulses=list(pulse_configs))
        new_src = DensityMatrixSource.from_pulse_ensemble(pulse_ensemble=ensemble, source_rate=source_rate, statistics_type=statistics_type, **kwargs)
        _copy_attrs(self, new_src)
    else:
        _orig_dm_init(self, *args, **kwargs)
DensityMatrixSource.__init__ = _patch_dm

# 3. Patch FiberChannel
_orig_ch_init = FiberChannel.__init__
def _patch_ch(self, *args, distance_km=None, fiber_loss_db_km=0.2, **kwargs):
    if distance_km is not None:
        tmp = FiberChannel.from_config_dict({"distance_km": distance_km, "fiber_loss_db_km": fiber_loss_db_km})
        _copy_attrs(self, tmp)
    else:
        _orig_ch_init(self, *args, **kwargs)
FiberChannel.__init__ = _patch_ch

# 4. Patch FiberChannel _transmittance alias (old scripts looked for private var)
if not hasattr(FiberChannel, '_transmittance') and hasattr(FiberChannel, 'transmittance'):
    FiberChannel._transmittance = property(lambda self: self.transmittance)

# 5. Patch SinglePhotonDetector dark_rate_d1
_orig_det_init = SinglePhotonDetector.__init__
def _patch_det(self, *args, dark_rate_d1=None, **kwargs):
    if dark_rate_d1 is None: dark_rate_d1 = 0.0
    _orig_det_init(self, *args, dark_rate_d1=dark_rate_d1, **kwargs)
SinglePhotonDetector.__init__ = _patch_det
