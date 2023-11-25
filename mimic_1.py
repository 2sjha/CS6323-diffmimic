from collections import defaultdict
import json as json
import numpy as np
import jax.numpy as jnp

def mimic(config_json):
    "Read provided config and set up Mimic training"

    mm_config = defaultdict()
    with open(config_json, 'r') as config_json_f:
        mm_config = json.load(config_json_f)
    
    ref_data = jnp

if __name__:
    mimic('config.json');