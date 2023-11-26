from .HUMANOID import _SYSTEM_CONFIG_HUMANOID

from google.protobuf import text_format
from brax.physics.config_pb2 import Config


def get_system_cfg(system_type):
    return {
      'humanoid': _SYSTEM_CONFIG_HUMANOID
    }[system_type]


def process_system_cfg(cfg):
    return text_format.Parse(cfg, Config())