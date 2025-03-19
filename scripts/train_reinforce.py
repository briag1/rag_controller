from src.agent.reinforce import Reinforce
from src.dataset.base import RAGDataset
from src.env.rag_env import RAGEnv
from src.env.rewarder.answer_F1 import RewarderF1
from src.trainer.base import Trainer

config = {
    "max_timesteps": 1000,
    "algo":"Reinforce",
    "max_time": 30 * 60,
    "episode_length": 2,
    "train_ep_returns":[],
    "eval_freq":500,
    "eval_episodes": 10,
    "save_filename":None,
    
}
agent = Reinforce(5, 1e-3, "answerdotai/ModernBERT-base", 0.9)
rag_dataset = RAGDataset.build_as_qa_dataset("data", agent.tokenizer)
rewarder = RewarderF1()
env = RAGEnv(rag_dataset, rewarder,"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Qwen/Qwen2.5-1.5B-Instruct")
trainer = Trainer()

trainer.train(env, agent, config, True)