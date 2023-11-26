import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as streamlit
from diffmimic.utils.io import deserialize_qp, serialize_qp
from diffmimic.mimic_envs import register_mimic_env

register_mimic_env()

streamlit.title('CS6323 - Project')
streamlit.subheader('Shubham Shekhar Jha (sxj220028)')
streamlit.subheader('Vedant Sapra (vks220000)')

def main():
    uploaded_file = streamlit.file_uploader("Evaluated Model Trajectory")
    if uploaded_file is not None:
        # show_rollout_traj(np.load(uploaded_file), 'DU')
        t = np.load(uploaded_file)
        print("t.shape=", t.shape, " t.shape[0]=", t.shape[0], " t.shape[1]=", t.shape[1])
        seed = streamlit.slider(f'Random seed ({"DU"})', 0, t.shape[1] - 1, 0)
        t = t[:, seed]

        rollout_qp = [deserialize_qp(t[i]) for i in range(t.shape[0])]
        t = serialize_qp(deserialize_qp(t))

        env = envs.get_environment(
            env_name="humanoid_mimic",
            system_config="humanoid",
            reference_traj=t,
        )
        components.html(html.render(env.sys, rollout_qp), height=500)

if __name__ == '__main__':
    main()
