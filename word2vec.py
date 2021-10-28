import numpy as np
import pandas as pd
import torch
from LargeMovieDataset import LargeMovieDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys

class Word2Vec(torch.nn.Module):

    def __init__(self,
                 input_voc,
                 embed_size=200):
        super(Word2Vec, self).__init__()

        self.dic = input_voc
        self.embed_size = embed_size
        self.voc_size = len(self.dic.keys())

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

        # First build the input vector
        x = x.reshape(-1, self.voc_size)
        # Apply the first layer
        x = self.layer_A(x)
        # If we want to get embedding, we can stop here
        if get_embed:
            return x
        # Predict context words probabilities
        x = self.layer_B(x)
        # Apply the softmax and return
        return self.sm(x)


def train_Word2Vec():

    # Hyperparameters
    learning_rate = 1e-5
    epoch = 20
    batch_size = 100
    model_name = 'word2vec'
    device = 'cpu'

    # Load datasets
    train_set = LargeMovieDataset(train=True, recover_serialized=True, device=device, output_mode='word2vec')
    test_set = LargeMovieDataset(train=False, recover_serialized=True, device=device, output_mode='word2vec')

    # Get data loaders
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size
    )
    test_loader = DataLoader(
        test_set,
        shuffle=True,
        batch_size=batch_size
    )

    # Instanciate the model
    model = Word2Vec(input_voc=train_set.dictionary)

    # The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_idx = 0
    for i in range(0, epoch):

        for step, (center_word, left_word, right_word) in enumerate(train_loader):

            print(center_word.shape, left_word.shape, right_word.shape)

            sys.exit(0)








        epoch_idx += 1




if __name__ == '__main__':

    train_Word2Vec()