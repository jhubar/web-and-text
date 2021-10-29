import numpy as np
import pandas as pd
import torch
import LargeMovieDataset
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


def train_Word2Vec():

    # Hyperparameters
    learning_rate = 1e-5
    epoch = 20
    batch_size = 15
    model_name = 'word2vec'
    model_path = 'G:/web_and_text_project/data/Large_movie_dataset/word2vec_model'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4


    # Load the dataset
    print('Dataset Loading...')
    data_builder = LargeMovieDataset.TrainSetBuilder(num_workers=num_workers, batch_size=batch_size)
    data_builder.import_cine_data()
    print('... Done')
    # Get data loaders
    train_loader, test_loader = data_builder.get_data_loader()
    # Get the tokenier object
    tokenizer = data_builder.get_tokenizer()

    # Instanciate the model
    model = Word2Vec(input_voc=data_builder.dictionary).to(device)

    # The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # The loss object
    loss_obj = torch.nn.NLLLoss()

    # Tensorboard object
    tb = SummaryWriter()

    epoch_idx = 0
    # Try to restore existing model if exists
    if not os.path.exists('{}/{}'.format(model_path, model_name)):
        os.mkdir('{}/{}'.format(model_path, model_name))
        file = open('{}/{}/train_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,train_loss\n')
        file.close()
        file = open('{}/{}/test_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,train_loss\n')
        file.close()
    else:
        # Try to load model's weights
        try:
            model.load_state_dict((torch.load('{}/{}/model_weights.pt'.format(model_path, model_name))))
            model_loaded = True
            print('Previous model loaded')
        except:
            print('Impossible to load existing model: start a new one')
        # Try to load optimizer weights
        try:
            optimizer.load_state_dict(torch.load('{}/{}/optimizer_weights.pt'.format(model_path, model_name)))
            print('Optimizer weights loaded')
        except:
            print('Fail to load optimizer weights')
        # Load epoch index if the model exists
        train_logs = pd.read_csv('{}/{}/train_logs.csv'.format(model_path, model_name), sep=',')
        epoch_idx = int(train_logs.iloc[-1]['Epoch']) + 1


    for i in range(epoch_idx, epoch):

        print('Taining epoch {} / {}'.format(i, epoch))
        loop = tqdm(train_loader, leave=True)

        model.train()
        train_loss = []
        for step, batch in enumerate(train_loader):

            # Get input ids (NOTE: batch[1] return attention mask and batch[2] return sentiments
            input_ids = torch.flatten(batch[0].to(device))

            # Extend with zero at the begin and at the end
            ext_input_ids = torch.zeros(input_ids.size(0) + 2).to(device)
            ext_input_ids[1:-1] = input_ids

            # makes predictions:
            preds = model(torch.tensor(input_ids))

            # Compute the loss for left and right context words
            loss = loss_obj(preds, ext_input_ids[0:-2].long()) + \
                   loss_obj(preds, ext_input_ids[2:].long())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Optimier step
            optimizer.step()

            # Store logs
            tb.add_scalar('Train_Loss_{}'.format(model_name), loss.item(), i)
            train_loss.append(loss.item())
            loop.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()

        # Write losses in logs
        f = open('{}/{}/train_logs.csv'.format(model_path, model_name), 'a')
        for itm in train_loss:
            f.write('{},{}\n'.format(i, itm))
        f.close()

        # Testing step
        model.eval()
        test_loss = []
        print('Testing epoch {} / {}'.format(i, epoch))
        loop = tqdm(test_loader, leave=True)
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                # Get input ids (NOTE: batch[1] return attention mask and batch[2] return sentiments
                input_ids = torch.flatten(batch[0].to(device))

                # Extend with zero at the begin and at the end
                ext_input_ids = torch.zeros(input_ids.size(0) + 2).to(device)
                ext_input_ids[1:-1] = input_ids

                # makes predictions:
                preds = model(torch.tensor(input_ids))

                # Compute the loss for left and right context words
                loss = loss_obj(preds, ext_input_ids[0:-2].long()) + \
                       loss_obj(preds, ext_input_ids[2:].long())

                # Store logs
                tb.add_scalar('Test_Loss_{}'.format(model_name), loss.item(), i)
                test_loss.append(loss.item())
                loop.set_postfix(loss=loss.item())
                torch.cuda.empty_cache()

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