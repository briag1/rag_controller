from abc import ABC, abstractmethod
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List
from typing import Any
import random


class Agent(ABC):
    
    def __init__(self)-> None:

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):

        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())
            
    @abstractmethod
    def act(self, obs: np.ndarray) -> Any:
        ...
    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        ...

    @abstractmethod
    def update(self) -> None:
        ...