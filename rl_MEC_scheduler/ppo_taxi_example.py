# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer

# Configure the algorithm.
config = {
            # Environment (RLlib understands openAI gym registered strings).
            "env": "Taxi-v3",
            # Use 2 environment workers (aka "rollout workers") that parallelly
            # collect samples from their own environment clone(s).
            "num_workers": 8,
            # Change this to "framework: torch", if you are using PyTorch.
            # Also, use "framework: tf2" for tf2.x eager execution.
            "framework": "tf",
            # Tweak the default model provided automatically by RLlib,
            # given the environment's observation- and action spaces.
            "model": {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
}

# Create our RLlib Trainer.
trainer = PPOTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
        print(trainer.train())


