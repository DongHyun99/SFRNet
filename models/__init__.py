import pickle
import types

import torch

from models.discriminator import NLayerDiscriminator
from models.ffc import FFCResNetGenerator

__all__ = ['FFCResNetGenerator', 'NLayerDiscriminator', 'build_generator', 'extract_generator_state_dict',
           'load_checkpoint_file', 'load_generator']


class _MissingClass:
    """Stand-in for classes pickled in third-party checkpoints whose package is not installed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_MissingClass,), {'__module__': module})


_tolerant_pickle = types.SimpleNamespace(Unpickler=_TolerantUnpickler, load=pickle.load, __name__='pickle')


def load_checkpoint_file(path, map_location='cpu'):
    """Loads any supported checkpoint. LaMa / PyTorch-Lightning checkpoints pickle Lightning and OmegaConf objects;
    those are replaced by stubs when the packages are not installed (only tensors are used)."""
    return torch.load(path, map_location=map_location, pickle_module=_tolerant_pickle, weights_only=False)


def build_generator():
    """SFR-Net generator (identical topology for the guided network G_S and the guiding network G_I)."""
    return FFCResNetGenerator(input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18)


def extract_generator_state_dict(checkpoint):
    """Returns the generator weights from an SFR-Net release file or a LaMa / PyTorch-Lightning checkpoint."""
    if 'generator' in checkpoint:
        return checkpoint['generator']
    if 'state_dict' in checkpoint:
        prefix = 'generator.'
        state = {k[len(prefix):]: v for k, v in checkpoint['state_dict'].items() if k.startswith(prefix)}
        if state:
            return state
    raise KeyError('No generator weights found in checkpoint (expected a "generator" or "state_dict" entry).')


def load_generator(path, map_location='cpu'):
    """Builds the generator and loads weights from ``path`` (strict)."""
    checkpoint = load_checkpoint_file(path, map_location=map_location)
    generator = build_generator()
    generator.load_state_dict(extract_generator_state_dict(checkpoint), strict=True)
    return generator
