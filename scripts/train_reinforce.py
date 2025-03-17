from src.agent.reinforce import Reinforce
from src.dataset.base import RAGDataset
from src.env.rag_env import RAGEnv
from src.env.rewarder.answer_F1 import RewarderF1

agent = Reinforce(5, 1e-3, "answerdotai/ModernBERT-base", 0.9)
rag_dataset = RAGDataset.build_as_qa_dataset("data", agent.tokenizer)
rewarder = RewarderF1()
env = RAGEnv(rag_dataset, rewarder,"answerdotai/ModernBERT-base", "Qwen/Qwen2.5-1.5B-Instruct")