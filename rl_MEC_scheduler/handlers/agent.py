import ray
from ray.rllib.agents import a3c
from ray.tune.registry import register_env

from rl_MEC_scheduler.handlers.env_handler import load_envs

ray.shutdown()

n_MEC = 5
n_UE = 10
w_mean = 1
w_max = 0

identifier = f"{n_MEC}_{n_UE}_{w_mean}_{w_max}"

env = load_envs(config_path="experiments/env_configs", config_filename=f"env_{identifier}_configs.json")[0]

register_env(
    "NetWorkEnv-v0",
    lambda config: env,
)

# Configure the algorithm.
config = {
            # Environment (RLlib understands openAI gym registered strings).
            "env": "NetWorkEnv-v0",
            # Use 2 environment workers (aka "rollout workers") that parallelly
            # collect samples from their own environment clone(s).
            "num_workers": 8,
            # Change this to "framework: torch", if you are using PyTorch.
            # Also, use "framework: tf2" for tf2.x eager execution.
            "framework": "tf",
            "horizon": 10
}

# Create our RLlib Trainer.
trainer = a3c.A2CTrainer(config=config)

for _ in range(300):
    result = trainer.train()
    chkpt_file = trainer.save(f"results/a2c_checkpoint_{identifier}")
    print(
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
    )
