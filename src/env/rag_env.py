
from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym
from src.dataset.base import RAGDataset
from src.model.rag.adaptative_top_k import AdaptiveTopKQueryEngine
from src.env.rewarder.base import Rewarder
class RAGEnv(gym.Env):
    def __init__(self, dataset: RAGDataset, rewarder: Rewarder, embedding_model: str, llm: str,):
        super().__init__()
        self.dataset = dataset.qa_dataset
        self.num_samples = len(self.dataset)
        # Define observation and action spaces
        max_question_length = max(len(sample["question"]) for sample in self.dataset)
        self.observation_space = gym.spaces.Text(max_length=max_question_length)
        self.action_space = gym.spaces.Discrete(4)  # top_k values: {1, 3, 5, 10, 20}
        self.rag = AdaptiveTopKQueryEngine.build(dataset.documents, embedding_model, llm)
        self.current_index = 0  # Random question index
        self.rewarder = rewarder

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_index = np.random.randint(0, self.num_samples)  # Random question index
        observation = self.dataset[self.current_index]
        return observation, {}


    def step(self, action):
        """Given an action (top_k selection), compute the reward based on retrieved documents."""
        question = self.dataset[self.current_index]["question"]
        reference_answer = self.dataset[self.current_index]["answer"]

        # Convert action index to top_k value: {0 → 1, 1 → 3, 2 → 5, 3 → 10, 4 → 20}
        top_k_values = [0, 1, 2, 3]
        top_k = top_k_values[action]

        # Get retrieved documents and compute relevance
        answer = self.rag.query_with_config(question, top_k)
        info = {"answer": answer, "reference_answer": reference_answer}
        # Compute reward
        reward, rewarder_info = self.rewarder.reward(answer, reference_answer)

        # Done flag (single-step environment)
        done = True

        return answer, reward, done, False, rewarder_info | info