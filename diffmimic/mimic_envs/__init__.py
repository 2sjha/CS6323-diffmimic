from brax import envs
from .mimic import Mimic
from .mimic_train import MimicTrain
from . import pd_controller


def setup_brax_envs():
    """Setup Brax Environments for Mimic Evaluation and Mimic Training"""
    envs.register_environment("mimic", Mimic)
    envs.register_environment("mimic_train", MimicTrain)
