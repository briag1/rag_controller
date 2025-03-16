import gymnasium as gym
from typing import Dict, Iterable, List
from src.agent.base import Agent
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import Adam

class Reinforce(Agent):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        encoder_name,
        gamma: float,
        **kwargs,
        )-> None:
        
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        num_actions = action_space.n
        
        self.encoder = AutoModelForSequenceClassification.from_pretrained(encoder_name, num_labels=num_actions, torch_dtype="auto")
        self.encoder_optim = Adam(self.encoder.parameters(), lr=learning_rate, eps=1e-3)


        self.learning_rate = learning_rate
        self.gamma = gamma

        self.saveables.update(
            {
                "policy": self.encoder,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int) -> None:
        pass

    def act(self, obs: str, explore: bool):
        self.policy

    def update(
        self, rewards: List[float], observations: List[str], actions: List[int],
        ) -> Dict[str, float]:
        current_return = 0
        output = self.policy(observations)
        for reward, obs, action in zip(reversed(rewards), reversed(observations), reversed(actions)):
            current_return = self.gamma * current_return + reward
            prob = [action]
        return {"p_loss": p_loss}