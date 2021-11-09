import torch
import numpy as np
import os
import sys
from word2vec import Word2Vec
from LargeMovieDataset import TrainSetBuilder
import torch.nn as nn

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
        #attention = self.sm(prdc / (self.embed_size ** (1/2)))
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

        # For attention step:
        self.att = SelfAttention()

        # A final layer to produce a sequence of values between 0 and 1

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

        # Apply self attention
        after_attention = self.att(hid_states, hid_states, hid_states)

        print(after_attention.shape)
        sys.exit(0)






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






