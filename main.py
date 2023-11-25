import functools
import logging
import json
import sys
from collections import defaultdict
import numpy as np
import jax.numpy as jnp
from brax import envs
from brax.io import metrics
from brax.training.agents.apg import networks as apg_networks
from diffmimic.mimic_envs.mimic import Mimic
from diffmimic.mimic_envs.mimic_train import MimicTrain
import diffmimic.brax_lib.agent_diffmimic as ag_dm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mimic(config_json):
    "Read provided config and set up Mimic training"

    # Register Brax enviroments
    envs.register_environment('mimic', Mimic)
    envs.register_environment('mimic_train', MimicTrain)

    mm_config = defaultdict()
    with open(config_json, 'r', encoding='utf-8') as config_json_f:
        mm_config = json.load(config_json_f)

    ref_data = jnp.array(np.load(mm_config['ref']))
    ref_len = ref_data.shape[0]

    # Set up brax environment parameters
    if ref_len < mm_config['ep_len']:
        mm_config['ep_len'] = ref_len
    if ref_len < mm_config['cycle_len']:
        mm_config['cycle_len'] = ref_len
    if ref_len < mm_config['ep_len_eval']:
        mm_config['ep_len_eval'] = ref_len

    mm_config['obs_type'] = 'timestamp'
    mm_config['early_termination'] = False
    mm_config['replay_rate'] = 0.05
    mm_config['truncation_length'] = None

    mimic_train_env = envs.get_environment(
        env_name="mimic_train",
        system_config=mm_config['system_config'],
        reference_traj=ref_data,
        obs_type=mm_config['obs_type'],
        cyc_len=mm_config['cycle_len'],
        total_length=mm_config['ep_len_eval'],
        rollout_length=mm_config['ep_len'],
        early_termination=mm_config['early_termination'],
        demo_replay_mode=mm_config['demo_replay_mode'],
        err_threshold=mm_config['threshold'],
        replay_rate=mm_config['replay_rate'],
        reward_scaling=mm_config['reward_scaling'],
        rot_weight=mm_config['rot_weight'],
        vel_weight=mm_config['vel_weight'],
        ang_weight=mm_config['ang_weight']
    )

    mimic_env = envs.get_environment(
        env_name="mimic",
        system_config=mm_config['system_config'],
        reference_traj=ref_data,
        obs_type=mm_config['obs_type'],
        cyc_len=mm_config['cycle_len'],
        rot_weight=mm_config['rot_weight'],
        vel_weight=mm_config['vel_weight'],
        ang_weight=mm_config['ang_weight']
    )

    log_writer = metrics.Writer('logs/')
    ag_dm.train(
        seed=mm_config['seed'],
        environment=mimic_train_env,
        eval_environment=mimic_env,
        episode_length=mm_config['ep_len']-1,
        eval_episode_length=mm_config['ep_len_eval']-1,
        num_envs=mm_config['num_envs'],
        num_eval_envs=mm_config['num_eval_envs'],
        learning_rate=mm_config['lr'],
        num_evals=mm_config['max_it']+1,
        max_gradient_norm=mm_config['max_grad_norm'],
        network_factory=functools.partial(
            apg_networks.make_apg_networks, hidden_layer_sizes=(512, 256)),
        normalize_observations=mm_config['normalize_observations'],
        save_dir='logs/',
        progress_fn=log_writer.write_scalars,
        use_linear_scheduler=mm_config['use_lr_scheduler'],
        truncation_length=mm_config['truncation_length'],
    )

if __name__:
    if len(sys.argv) > 2:
        print('Usage: python mimic.py <config.json>', file=sys.stderr)
    elif len(sys.argv) == 1:
        print('Running default config.json')
        mimic('config.json')
    else:
        mimic(sys.argv[1])
