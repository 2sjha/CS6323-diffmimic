import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as streamlit
from diffmimic.utils.io import deserialize_qp, serialize_qp
from diffmimic.mimic_envs import setup_brax_envs
from diffmimic.mimic_envs.humanoid_system_config import Humanoid_System_Config

setup_brax_envs()

<<<<<<< HEAD
st.title('DiffMimic - Visualization')
with st.expander("Readme"):
    st.markdown(
        'Input modes:'
        '\n- File Path: input the local file path to the .npy trajectory file.'
        '\n- Experiment Log: select trajectories by browsing experiment logs.'
        '\n- Direct Upload: directly upload the .npy trajectory file.')


def show_rollout_traj(rollout_traj, tag):
    if len(rollout_traj.shape) == 3:
        seed = st.slider(f'Random seed ({tag})', 0, rollout_traj.shape[1] - 1, 0)
        rollout_traj = rollout_traj[:, seed]

    rollout_qp = [deserialize_qp(rollout_traj[i]) for i in range(rollout_traj.shape[0])]
    rollout_traj = serialize_qp(deserialize_qp(rollout_traj))

    env = envs.get_environment(
        env_name="mimic",
        system_config=Humanoid_System_Config,
        reference_traj=rollout_traj,
        obs_type='timestamp',
        cyc_len=None,
        reward_scaling=1.,
        rot_weight=1.,
        vel_weight=0.,
        ang_weight=0.
    )
    components.html(html.render(env.sys, rollout_qp), height=500)

=======
streamlit.title('CS6323 - Project')
streamlit.subheader('Shubham Shekhar Jha (sxj220028)')
streamlit.subheader('Vedant Sapra (vks220000)')
>>>>>>> main

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
