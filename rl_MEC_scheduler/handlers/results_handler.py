import os

import gym

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rl_MEC_scheduler.handlers.baselines import get_baselines
from rl_MEC_scheduler.handlers.env_handler import load_envs, NetworkEnv
from rl_MEC_scheduler.services.plot_services import add_plot
from rl_MEC_scheduler.services.location_services import get_locations, loc_to_df

n_MEC = int(os.environ["n_MEC"])
n_UE = int(os.environ["n_UE"])
w_mean = int(os.environ["w_mean"])
w_max = int(os.environ["w_max"])

identifier = f"{n_MEC}_{n_UE}_{w_mean}_{w_max}_heterogenous"

network_env = load_envs(
    config_path="experiments/env_configs",
    config_filename=f"env_{identifier}_configs.json",
)[0]

gym.register(
    id="NetworkEnv-v0",
    entry_point=NetworkEnv,
    max_episode_steps=10,
)

gym_env = gym.make(
    "NetworkEnv-v0",
    seed_value=network_env.seed_value,
    UEs=network_env.UEs,
    MECs=network_env.MECs,
    network=network_env.network,
    task_distributions=network_env.task_distributions,
    mean_weight=w_mean,
    max_weight=w_max,
)

baselines = get_baselines(gym_env, n_episodes=100, n_steps=10)  # type: ignore
print("Local:", baselines[0][1])
print("Random Agent:", baselines[2][1])
print("Closest MEC:", baselines[1][1])

df = pd.read_csv(f"results/{identifier}/progress.csv")

fig = go.Figure()

x = list(df["timesteps_total"])

palette = [
    "38, 70, 83",
    "42, 157, 143",
    "233, 196, 106",
    "244, 162, 97",
    "231, 111, 81"
]

fig = add_plot(
    fig=fig,
    x=x,
    y_lower=list(),
    y=list(df["episode_reward_mean"]),
    y_upper=list(),
    label="RL Agent",
    rgb_str=palette[0]
)

fig = add_plot(
    fig=fig,
    x=x,
    y_lower=[],
    y=[baselines[1][1]] * len(x),
    y_upper=[],
    label="Closest MEC",
    rgb_str=palette[2]
)

fig = add_plot(
    fig=fig,
    x=x,
    y_lower=[],
    y=[baselines[2][1]] * len(x),
    y_upper=[],
    label="Random Agent",
    rgb_str=palette[3]
)

fig.update_layout(
    xaxis_title="Timesteps",
    yaxis_title="Reward",
    xaxis_range=[0, 450000],
    width=600,
    height=500,
)
fig.update_traces(mode="lines")
fig.show()

UE_locs = get_locations(network_env.UEs)
MEC_locs = get_locations(network_env.MECs)

df_UEs = loc_to_df(locations=UE_locs, type="UE", n_dimensions=2)
df_MECs = loc_to_df(locations=MEC_locs, type="MEC", n_dimensions=2)

df_locs = pd.concat([df_UEs, df_MECs], ignore_index=True)

fig = px.scatter(df_locs, x="x", y="y", color="type", width=500, height=500)
fig.show()
