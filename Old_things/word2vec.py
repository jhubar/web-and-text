import numpy as np
import pandas as pd
import torch
from LargeMovieDataset import LargeMovieDataset
from LargeMovieDataset import word2vec_collate_fn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

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
    batch_size = 1
    model_name = 'word2vec'
    model_path = 'G:/web_and_text_project/data/Large_movie_dataset/word2vec_model'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # Load datasets
    train_set = LargeMovieDataset(train=True, recover_serialized=True, device=device, output_mode='word2vec')
    test_set = LargeMovieDataset(train=False, recover_serialized=True, device=device, output_mode='word2vec')

    # Get data loaders
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=word2vec_collate_fn,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=word2vec_collate_fn,
        num_workers=num_workers
    )

    # Instanciate the model
    model = Word2Vec(input_voc=train_set.dictionary).to(device)

    # The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # The loss object
    loss_obj = torch.nn.NLLLoss()

    # Tensorboard object
    tb = SummaryWriter()

    # Try to restore existing model if exists
    model_loaded = False
    try:
        model.load_state_dict((torch.load('{}/{}/model_weights.pt'.format(model_path, model_name))))
        model_loaded = True
        print('Previous model loaded')
    except:
        print('Impossible to load existing model: start a new one')
    # Try to load optimizer weights
    if model_loaded:
        try:
            optimizer.load_state_dict(torch.load('{}/{}/optimizer_weights.pt'.format(model_path, model_name)))
            print('Optimizer weights loaded')
        except:
            print('Fail to load optimizer weights')

    epoch_idx = 0
    # Try to load training epoch index using logs
    if model_loaded:
        if os.path.exists('{}/{}/train_logs.csv'):
            train_logs = pd.read_csv('{}/{}/train_logs.csv'.format(model_path, model_name), sep=',')
            epoch_idx = train_logs[-1, 0] + 1

    # Create new logs file if doesn't exist
    if not model_loaded:
        os.mkdir('{}/{}'.format(model_path, model_name))
        file = open('{}/{}/train_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,train_loss\n')
        file.close()
        file = open('{}/{}/test_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,train_loss\n')
        file.close()



    for i in range(epoch_idx, epoch):

        print('Taining epoch {} / {}'.format(epoch_idx, epoch))
        loop = tqdm(train_loader, leave=True)

        model.train()
        train_loss = []
        for step, (center_word, context_words) in enumerate(train_loader):

            center_word = center_word.to(device)
            context_words = context_words.to(device)
            # makes predictions:
            preds = model(torch.tensor(center_word))

            # Compute the loss
            loss = loss_obj(preds, torch.flatten(context_words.long()))

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Optimier step
            optimizer.step()

            # Store logs
            tb.add_scalar('Train_Loss_{}'.format(model_name), loss.item(), i)
            train_loss.append(loss.item())
            loop.set_postfix(loss=loss.item)
            torch.cuda.empty_cache()

        # Write losses in logs
        f = open('{}/{}/train_logs.csv'.format(model_path, model_name), 'a')
        for itm in train_loss:
            f.write('{},{}\n'.format(i, itm))
        f.close()

        # Testing step
        model.eval()
        test_loss = []
        print('Testing epoch {} / {}'.format(epoch_idx, epoch))
        loop = tqdm(test_loader, leave=True)
        for step, (center_word, context_words) in enumerate(test_loader):
            with torch.no_grad():
                center_word = center_word.to(device)
                context_words = context_words.to(device)
                # makes predictions:
                preds = model(torch.tensor(center_word))

                # Compute the loss
                loss = loss_obj(preds, torch.flatten(context_words.long()))

                # Store logs
                tb.add_scalar('Test_Loss_{}'.format(model_name), loss.item(), i)
                test_loss.append(loss.item())
                loop.set_postfix(loss=loss.item)

        # Write losses in logs
        f = open('{}/{}/test_logs.csv'.format(model_path, model_name), 'a')
        for itm in test_loss:
            f.write('{},{}\n'.format(i, itm))
        f.close()

        # Save the model
        print('Model saving...')
        torch.save(model.state_dict(), '{}/{}/model_weights.pt'.format(model_path, model_name))
        torch.save(optimizer.state_dict(), '{}/{}/optimizer_weights.pt'.format(model_path, model_name))
        print('... Done.')




if __name__ == '__main__':

    train_Word2Vec()