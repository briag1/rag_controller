import gymnasium as gym
from typing import Dict, Iterable, List
from src.agent.base import Agent
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.optim import Adam

class Reinforce(Agent):

    def __init__(
        self,
        num_actions: int,
        learning_rate: float,
        encoder_name,
        gamma: float,
        **kwargs,
        )-> None:

        
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
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

    def act(self, obs: dict, explore: bool):
        with torch.no_grad:
            logits: torch.Tensor = self.encoder(input_ids = obs["input_ids"], attention_mask = obs["attention_mask"]).logits
        if explore:
            indices = np.arange(logits)
            
        logits.softmax(dim = 1)

    def update(
        self, rewards: List[float], observations: List[str], actions: List[int],
        ) -> Dict[str, float]:
        current_return = 0
        output = self.policy(observations)
        for reward, obs, action in zip(reversed(rewards), reversed(observations), reversed(actions)):
            current_return = self.gamma * current_return + reward
            prob = [action]
        return {"p_loss": p_loss}