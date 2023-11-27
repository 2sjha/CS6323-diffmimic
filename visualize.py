import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as streamlit
from diffmimic.utils.io import deserialize_qp, serialize_qp
from diffmimic.mimic_envs import setup_brax_envs
from diffmimic.mimic_envs.humanoid_system_config import Humanoid_System_Config

setup_brax_envs()

streamlit.title('CS6323 - Project')
streamlit.subheader('Shubham Shekhar Jha (sxj220028)')
streamlit.subheader('Vedant Sapra (vks220000)')

def main():
    uploaded_file = streamlit.file_uploader("Evaluated Model Trajectory")
    if uploaded_file is not None:
        t = np.load(uploaded_file)
        t = t[:, 0]

        rollout_qp = [deserialize_qp(t[i]) for i in range(t.shape[0])]
        t = serialize_qp(deserialize_qp(t))

        env = envs.get_environment(env_name="mimic",system_config=Humanoid_System_Config,reference_traj=t,)
        components.html(html.render(env.sys, rollout_qp), height=500)

if __name__ == '__main__':
    main()
