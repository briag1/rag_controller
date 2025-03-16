
from torch import nn, Tensor
from typing import Iterable
# Load model directly
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch

class EncoderForClassification(nn.Module):
    
    def __init__(self,num_actions:int, output_activation: nn.Module = None, encoder_name: str= "sentence-transformers/paraphrase-MiniLM-L12-v2"):
        super().__init__()
        

        self.encoder = AutoModelForSequenceClassification.from_pretrained(encoder_name, num_labels=num_actions, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)                                                                                    
        
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Feedforward
        return self.encoder(input_ids = inputs)

    def hard_update(self, source: nn.Module):
        """Update the network parameters by copying the parameters of another network.

        :param source (nn.Module): network to copy the parameters from
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source: nn.Module, tau: float):
        """Update the network parameters with a soft update.

        Moves the parameters towards the parameters of another network

        :param source (nn.Module): network to move the parameters towards
        :param tau (float): stepsize for the soft update
            (tau = 0: no update; tau = 1: copy parameters of source network)
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )
    