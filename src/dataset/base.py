from __future__ import annotations
import json
import pathlib


from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional
from datasets import Dataset

class RAGDataset:
    documents: list[str] 
    qa_dataset: Dataset
    def __init__(self, qa_dataset: Dataset, documents: list[str])-> None:
        self.qa_dataset = qa_dataset
        self.documents = documents
    @classmethod
    def build_rag_dataset(cls, dataset_name: str, path_raw_data: str)-> RAGDataset:
        match dataset_name:
            case "asqa":
                return cls.build_as_qa_dataset()
            case _:
                raise ValueError(dataset_name)
            
    @classmethod      
    def build_as_qa_dataset(cls, path_raw_data: str)->RAGDataset:
        if path_raw_data is None: 
            cls.download_alce_data(path_raw_data)
        with pathlib.Path("{}/asqa_eval_gtr_top100_reranked_oracle.json".format(path_raw_data)).open() as data_file:
            data = json.load(data_file)
        dataset = Dataset.from_list(data)
        documents_texts_dict = {}
        for documents in dataset["docs"]:
            for doc in documents:
                if doc["id"] not in documents_texts_dict:
                    documents_texts_dict[doc["id"]] = doc["text"]
        def add_answers(sample):
            answers = [pair["short_answers"] for pair in sample["qa_pairs"]]
            return {"answer": answers}
        new_ds = dataset.map(add_answers, remove_columns=["qa_pairs", "wikipages","annotations","sample_id", "docs"])
        return RAGDataset(documents = documents_texts_dict, qa_dataset=new_ds)
    @classmethod
    def download_alce_data(cls, path_raw_data):
        raise NotImplementedError()
        # wget https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar
        # tar xvf ALCE-data.tar
        # mv ALCE-data data
        # !echo "deleting tar file..."
        # !rm ALCE-data.tar
        
            
        
        
    