import gymnasium as gym
from typing import Dict, Iterable, List

class Reinforce(Agent):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
        )-> None:
        
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
            )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)


        self.learning_rate = learning_rate
        self.gamma = gamma

        self.saveables.update(
            {
                "policy": self.policy,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int) -> None:
        pass

    def act(self, obs: np.ndarray, explore: bool):
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q3")

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q3")
        p_loss = 0.0
        return {"p_loss": p_loss}