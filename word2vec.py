import numpy as np
import pandas as pd
import torch
from LargeMovieDataset import LargeMovieDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class Word2Vec(torch.nn.Module):

    def __init__(self,
                 input_voc,
                 embed_size=200):

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
    train_set = LargeMovieDataset(train=True, recover_serialized=True)
    test_set = LargeMovieDataset(train=False, recover_serialized=True)

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

        for step, (sentences, sentiments) in enumerate(train_loader):

            for seq in sentences:
                # Get one hot encoding for the sentence
                input_one_hot = torch.zeros((len(seq), model.voc_size)).to(device)
                for j in range(0, len(seq)):
                    input_one_hot[j, seq[j]] = 1

                # Make prediction for the sentence
                preds = model(input_one_hot)







        epoch_idx += 1




