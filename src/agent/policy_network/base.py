
from torch import nn, Tensor
from typing import Iterable
# Load model directly
from transformers import AutoTokenizer, AutoModel


import torch

class FCNEncoder(nn.Module):
    
    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None, encoder_name: str= "sentence-transformers/paraphrase-MiniLM-L12-v2"):
        """Create a network using ReLUs between layers and no activation at the end.

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        """
        super().__init__()
        self.input_size = dims[0]
        self.out_size = dims[-1]
        self.layers = self.make_seq(dims, output_activation)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
    @staticmethod
    def make_seq(dims: Iterable[int], output_activation: nn.Module) -> nn.Module:
        """Create a sequential network using ReLUs between layers and no activation at the end.

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :return (nn.Module): return created sequential layers
        """
        mods = []

        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.ReLU())

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        """Compute a forward pass through the network.

        :param x (torch.Tensor): input tensor to feed into the network
        :return (torch.Tensor): output computed by the network
        """
        # Feedforward
        return self.layers(x)

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
    