
from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.model.text_2_text_models.llm_agent import LLMAgent


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Answer factoid (answer can be short i.e keyword only) and ambiguous query below:\n"
    "{query_str}\n"
    "Given the following context information and not prior knowledge: "
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Since the query is ambiguous you must give all the possible keyword answers on a separate line."
    "Since the question is factoid and you answer MUST only contain a few keywords."
)


class AdaptiveTopKQueryEngine(CustomQueryEngine):

  retriever: BaseRetriever
  llm: LLMAgent

  def set_top_k(self, top_k: int) -> None:
    self._top_k = top_k

  def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
      print("top_k: ",self._top_k)
      nodes = self.retriever.retrieve(query_str)[:self._top_k]
      prompt =DEFAULT_TEXT_QA_PROMPT_TMPL.format(query_str=query_str, context_str="\n".join([node.text for node in nodes]))
      response = self.llm.query(prompt)
      return response

  def query_with_config(self, query_str: str, top_k: int) -> STR_OR_RESPONSE_TYPE:
    self.set_top_k(top_k)
    return self.custom_query(query_str)

  @classmethod
  def build(cls, documents_texts_dict: dict[int,str], embedding_model_name: str, llm_name: str)-> AdaptiveTopKQueryEngine:
    embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name, embed_batch_size=64)
    index = VectorStoreIndex.from_documents([Document(text = doc) for doc in documents_texts_dict.values()], embed_model= embedding_model)
    retriever = VectorIndexRetriever(
          index=index,
          embed_model=embedding_model,
          similarity_top_k=50)

    agent = LLMAgent(llm_name)
    rag = AdaptiveTopKQueryEngine(
          retriever=retriever,
          llm=agent
      )
    return rag