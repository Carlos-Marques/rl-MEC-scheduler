from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np


@dataclass
class ReplayBuffer:
    mem_size: int
    input_shape: int
    n_actions: int
    discrete: bool = False
    mem_cntr: int = 0

    state_memory: np.ndarray = field(init=False)
    new_state_memory: np.ndarray = field(init=False)
    action_memory: np.ndarray = field(init=False)
    reward_memory: np.ndarray = field(init=False)
    terminal_memory: np.ndarray = field(init=False)

    def __post__init__(self):
        self.state_memory = np.zeros((self.mem_size, self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, self.input_shape))

        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=dtype)

        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)


def store_transition(
    replay_buff: ReplayBuffer,
    state: np.ndarray,
    action: Union[int, np.ndarray],
    reward: float,
    new_state: np.ndarray,
    done: float,
) -> ReplayBuffer:
    index = replay_buff.mem_cntr % replay_buff.mem_size

    replay_buff.state_memory[index] = state
    replay_buff.new_state_memory[index] = new_state

    if replay_buff.discrete:
        actions = np.zeros(replay_buff.n_actions)
        actions[action] = 1.0
        replay_buff.action_memory[index] = actions
    else:
        replay_buff.action_memory[index] = action

    replay_buff.reward_memory[index] = reward
    replay_buff.terminal_memory[index] = 1 - done

    replay_buff.mem_cntr += 1

    return replay_buff


def sample_buffer(
    replay_buff: ReplayBuffer, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_mem = min(replay_buff.mem_cntr, replay_buff.mem_size)
    batch = np.random.choice(max_mem, batch_size)

    states = replay_buff.state_memory[batch]
    actions = replay_buff.action_memory[batch]
    rewards = replay_buff.reward_memory[batch]
    new_states = replay_buff.new_state_memory[batch]
    terminal = replay_buff.terminal_memory[batch]

    return states, actions, rewards, new_states, terminal
