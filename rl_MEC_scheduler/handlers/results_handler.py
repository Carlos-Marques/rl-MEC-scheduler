import gym

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rl_MEC_scheduler.handlers.baselines import get_baselines
from rl_MEC_scheduler.handlers.env_handler import load_envs, NetworkEnv
from rl_MEC_scheduler.services.plot_services import add_plot
from rl_MEC_scheduler.services.location_services import get_locations, loc_to_df

n_MECs = 5
n_UEs = 20
xp_name = "A2CTrainer_NetWorkEnv-v0_2022-05-09_14-30-11eim4m1a8"

network_env = load_envs(
    config_path="experiments/env_configs",
    config_filename=f"env_{n_MECs}_{n_UEs}_configs.json",
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
)

baselines = get_baselines(gym_env, n_episodes=100, n_steps=10)  # type: ignore
print(
    "Local:", baselines[0],
    "Closest MEC:", baselines[1],
    "Random Agent:", baselines[2],
)

df = pd.read_csv(f"results/ray_results/{xp_name}/progress.csv")

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
    y_lower=list(df["episode_reward_min"]),
    y=list(df["episode_reward_mean"]),
    y_upper=list(df["episode_reward_max"]),
    label="RL Agent",
    rgb_str=palette[0]
)

fig = add_plot(
    fig=fig,
    x=x,
    y_lower=[baselines[1][0]] * len(x),
    y=[baselines[1][1]] * len(x),
    y_upper=[baselines[1][2]] * len(x),
    label="Closest MEC",
    rgb_str=palette[2]
)

fig = add_plot(
    fig=fig,
    x=x,
    y_lower=[baselines[2][0]] * len(x),
    y=[baselines[2][1]] * len(x),
    y_upper=[baselines[2][2]] * len(x),
    label="Random Agent",
    rgb_str=palette[3]
)

fig.update_layout(
    title=f"Agent's Performance while training - MECs: {n_MECs} UEs: {n_UEs}",
    xaxis_title="Timesteps",
    yaxis_title="Reward",
)
fig.update_traces(mode="lines")
fig.show()

UE_locs = get_locations(network_env.UEs)
MEC_locs = get_locations(network_env.MECs)

df_UEs = loc_to_df(locations=UE_locs, type="UE", n_dimensions=2)
df_MECs = loc_to_df(locations=MEC_locs, type="MEC", n_dimensions=2)

df_locs = pd.concat([df_UEs, df_MECs], ignore_index=True)

fig = px.scatter(df_locs, x="x", y="y", color="type", title=f"Locations - MECs: {n_MECs} UEs: {n_UEs}", width=500, height=500)
fig.show()
