import numpy as np
import pandas as pd
import torch
import LargeMovieDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


class FastText(torch.nn.Module):
    def __init__(self, input_voc, embed_size=200):
        super(FastText, self).__init__()
        self.dic = input_voc
        self.embed_size = embed_size
        self.voc_size = len(self.self.dict.keys())
        # Input layer: must produce the embedding
        self.layer_A = torch.nn.Linear(in_features=self.voc_size,
                                       out_features=self.embed_size)

        # Output layer: must return probabilities for all words in dict
        self.layer_B = torch.nn.Linear(in_features=self.embed_size,
                                       out_features=self.voc_size)
        # The softmax of the output
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x, get_embed=False):
        """
        Input must be one hot one hot encoding vectors of the size of the
        given dictionary
        """

        # Transform index sequence in one hot encoding
        x = torch.nn.functional.one_hot(x, num_classes=self.voc_size)
        # First build the input vector
        x = x.reshape(-1, self.voc_size)
        # Apply the first layer
        x = self.layer_A(x.float())
        # If we want to get embedding, we can stop here
        if get_embed:
            return x
        # Predict context words probabilities
        x = self.layer_B(x)
        # Apply the softmax and return
        return self.sm(x)


def train_FastText():
    pass


if __name__ == '__main__':
    trainTastText()
