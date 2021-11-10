import torch
import numpy as np
import os
import sys
from word2vec import Word2Vec
from LargeMovieDataset import TrainSetBuilder
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size=50, max_len=400):
        super(PositionalEncoding, self).__init__()

        self.embed_size = embed_size
        self.max_len = max_len

        # Store a matrix with all possible positions
        pe = torch.zeros(embed_size, max_len)
        for pos in range(0, max_len):
            for i in range(0, embed_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[i + 1, pos] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # Get seq size
        seq_len = x.size(2)

        # If size is greater that pos embedding saved in memory:
        if seq_len > self.max_len:
            self.adapt_len(seq_len)

        # Add positional embedding
        x = x[:, 0:self.embed_size, 0:seq_len] + self.pe[:, :seq_len].to('cuda:0')
        return x

    def adapt_len(self, new_len):

        self.max_len = new_len

        # Store a matrix with all possible positions
        pe = torch.zeros(self.embed_size, self.max_len)
        for pos in range(0, self.max_len):
            for i in range(0, self.embed_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / self.embed_size)))
                pe[i + 1, pos] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


class SelfAttention(nn.Module):
    """
    The self attetion module
    """

    def __init__(self, embed_size=2048, nb_heads=4):
        """
        :param embed_size: This size is the kernel size of the enbedding
        convolutional layer.
        :param nb_heads: The number of heads in the self attention process
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        # WARNING: the embed_size have to be a multiple of the number of heads
        self.nb_heads = nb_heads
        self.heads_dim = int(embed_size / nb_heads)

        # Layer to generate the values matrix
        self.fc_values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Layer to generate keys
        self.fc_keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Layer for Queries
        self.fc_queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        # A fully connected layer to concatenate results
        self.fc_concat = nn.Linear(self.nb_heads * self.heads_dim, embed_size)

        # The softmax step
        self.sm = nn.Softmax(dim=3)

    def forward(self, values, keys, query, mask=None):
        # Get the number of training samples
        n = query.shape[0]
        # Get original shapes
        v_len = values.size()[1]
        k_len = keys.size()[1]
        q_len = keys.size()[1]

        # Split embedded inputs into the number of heads
        v = values.view(n, v_len, self.nb_heads, self.heads_dim)
        k = keys.view(n, k_len, self.nb_heads, self.heads_dim)
        q = query.view(n, q_len, self.nb_heads, self.heads_dim)

        # Feed it in appropriate layer
        v = self.fc_values(v)
        k = self.fc_keys(k)
        q = self.fc_queries(q)

        # Matrix dot product between qureries and keys
        prdc = torch.einsum('nqhd,nkhd->nhqk', [q, k])

        # Apply mask if present
        if mask is not None:
            prdc = prdc.masked_fill(mask == 0, float('-1e20'))  # don't use zero

        # The softmax step
        # attention = self.sm(prdc / (self.embed_size ** (1/2)))
        attention = torch.softmax(prdc / (self.embed_size ** (1 / 2)), dim=3)

        # Product with values
        # Output shape: (n, query len, heads, head_dim
        out = torch.einsum('nhql,nlhd->nqhd', [attention, v])

        # Concatenate heads results (n x query len x embed_size)
        out = torch.reshape(out, (n, q_len, self.nb_heads * self.heads_dim))

        # Feed the last layer
        return self.fc_concat(out)


class SentimentModel(torch.nn.Module):

    def __init__(self,
                 embed_size=200,
                 name='Test',
                 device='cpu',
                 embedding=None,
                 sentences_length=500):
        super(SentimentModel, self).__init__()
        self.name = name
        self.embed_size = embed_size
        self.device = device
        self.hidden_size = 1024
        self.sentences_length = sentences_length
        self.nb_heads = 4  # Number of heads for multi head self attention

        # Store the embedding layer: the model have to be given in parameters
        self.embed_layer = embedding
        # Don't train this model
        self.embed_layer.eval()

        # Bi directional LSTM layer
        self.rnn_1 = torch.nn.LSTM(embed_size,
                                   self.hidden_size,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True)

        # Positional encoding to apply before the self attention
        self.pos_encod = PositionalEncoding(embed_size=self.hidden_size * 2,
                                            max_len=self.sentences_length)
        # For attention step:
        self.att = SelfAttention(embed_size=self.hidden_size * 2,
                                 nb_heads=self.nb_heads)

        # Classify each word with a value between -1 and 1
        self.word_fc = torch.nn.Linear(in_features=self.hidden_size * 2,
                                       out_features=1)
        # Tanh activation for this layer
        self.word_sig = torch.nn.Sigmoid()

    def forward(self, x):
        # Get the embedding of inputs
        with torch.no_grad():
            embed_x = self.embed_layer.forward(x, get_embed=True)

        # Apply the bidirectional LSTM
        hid_states, (final_h_states, final_c_state) = self.rnn_1(embed_x)
        # WARNING shapes:
        #   For hid_state: Batch - seq length - 2 * hidden size
        #   For final states: 2 - batch_size - hidden size
        if True:
            print('hidden_states: {}'.format(hid_states.shape))
            print('final_hidden_state: {}'.format(final_h_states.shape))
            print('final_cell_state: {}'.format(final_c_state.shape))

        # Apply positional encoding:
        hid_states = self.pos_encod(hid_states)
        # Apply self attention
        after_attention = self.att(hid_states, hid_states, hid_states)

        # Apply fully connected layer sigmoid activation
        words_values = self.word_fc(after_attention)
        words_values = self.word_sig(words_values)

        # Do the mean
        outputs = torch.mean(words_values, dim=-1)

        return outputs, words_values


if __name__ == '__main__':

    # Embedding model math
    embed_model_path = 'D:/web_and_text_project/data/Large_movie_dataset/word2vec_model'

    # Model parameters
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    num_workers = 4
    embed_size = 100
    embed_name = 'word2vec'
    batch_size = 5

    # Load the dataset
    print('Dataset Loading...')
    data_builder = TrainSetBuilder(num_workers=num_workers, batch_size=batch_size)
    data_builder.import_cine_data()
    print('... Done')

    # Get data loaders
    train_loader, test_loader = data_builder.get_data_loader()

    # Load the embedding model
    embedder = Word2Vec(input_voc=data_builder.dictionary,
                        embed_size=embed_size).to(device)

    try:
        embedder.load_state_dict((torch.load('{}/{}/model_weights.pt'.format(embed_model_path, embed_name),
                                             map_location=device)))
        print('Embedding weights loaded')
    except:
        print('Impossible to load Embedding weights')
        sys.exit(1)

    # Instanciate the model
    engine = SentimentModel(embed_size=embed_size,
                            name='TestA',
                            device=device,
                            embedding=embedder)

    # Reading loop
    for step, batch in enumerate(train_loader):
        # Get one-hot indexes vector
        input_ids = batch[0]

        preds = engine(input_ids)
