from brax import envs
from .humanoid_mimic import Mimic
from .humanoid_mimic_train import MimicTrain
from . import pd_controller


def register_mimic_env():
    envs.register_environment('mimic', Mimic)
    envs.register_environment('mimic_train', MimicTrain)
