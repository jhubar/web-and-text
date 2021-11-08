import torch
import numpy as np
import os
import sys
from word2vec import Word2Vec
from LargeMovieDataset import TrainSetBuilder

class SentimentModel(torch.nn.Module):

    def __init__(self,
                 embed_size=200,
                 name='Test',
                 device='cpu',
                 embedding=None):

        super(SentimentModel, self).__init__()
        self.name = name
        self.embed_size = embed_size
        self.device = device
        self.hidden_size = 1024

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

        # Fully connected layers to apply

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

        sys.exit(0)







if __name__ == '__main__':

    # Embedding model math
    embed_model_path = 'D:/web_and_text_project/data/Large_movie_dataset/word2vec_model'

    # Model parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    embed_size = 200
    embed_name = 'word2vec_xavier'
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
        embedder.load_state_dict(torch.load(torch.load('{}/{}/model_weights.pt'.format(embed_model_path, embed_name),
                                                        map_location=device)))
        print('Embedding weights loaded')
    except:
        print('Impossible to load Embedding weights')
        #sys.exit(1)

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






