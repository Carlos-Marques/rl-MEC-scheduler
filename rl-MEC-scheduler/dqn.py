from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import adam_v2

from replay_buffer import ReplayBuffer, store_transition, sample_buffer


def build_model(
    lr: float, n_actions: int, input_dims: int, fc1_dims: int, fc2_dims: int
) -> Sequential:
    model = Sequential(
        [
            Dense(fc1_dims, input_shape=(input_dims,)),
            Activation("relu"),
            Dense(fc2_dims),
            Activation("relu"),
            Dense(n_actions),
        ]
    )

    model.compile(optimizer=adam_v2.Adam(lr=lr), loss="mse")

    return model


@dataclass
class Agent:
    alpha: float
    gamma: float
    n_actions: int
    epsilon: float
    batch_size: int
    input_dims: int
    epsilon_dec: float = 0.996
    epsilon_end: float = 0.01
    mem_size: int = 1000000
    fname: str = "dqn_model.h5"

    memory: ReplayBuffer = field(init=False)
    q_eval: Sequential = field(init=False)

    def __post__init__(self):
        self.memory = ReplayBuffer(
            self.mem_size, self.input_dims, self.n_actions, discrete=True
        )
        self.q_eval = build_model(self.alpha, self.n_actions, self.input_dims, 256, 256)


def remember(
    agent: Agent,
    state: np.ndarray,
    action: int,
    reward: float,
    new_state: np.ndarray,
    done: bool,
) -> Agent:
    agent.memory = store_transition(
        agent.memory, state, action, reward, new_state, done
    )

    return agent


def choose_action(agent: Agent, state: np.ndarray) -> Union[int, np.intp]:
    if np.random.random() > agent.epsilon:
        actions = agent.q_eval.predict(state)
        action = np.argmax(actions)
    else:
        action = np.random.choice(agent.n_actions)

    return action


def learn(agent: Agent) -> Agent:
    if agent.memory.mem_cntr > agent.batch_size:
        state, action, reward, new_state, done = sample_buffer(
            agent.memory, agent.batch_size
        )

        action_values = np.arange(agent.n_actions, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = agent.q_eval.predict(state)
        q_next = agent.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(agent.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = (
            reward + agent.gamma * np.max(q_next, axis=1) * done
        )

        agent.q_eval.fit(state, q_target, verbose="silent")

        agent.epsilon = (
            agent.epsilon * agent.epsilon_dec
            if agent.epsilon > agent.epsilon_end
            else agent.epsilon_end
        )

    return agent


def save_agent_model(agent: Agent) -> None:
    agent.q_eval.save(agent.fname)


def load_agent_model(agent: Agent) -> Optional[Agent]:
    model = load_model(agent.fname)
    if model:
        agent.q_eval = model
        return agent
    
    return None
