import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import sys
import nltk         # For word tokenization
import pickle
from nltk.tokenize.punkt import PunktBaseClass
nltk.download('punkt')
import torch

class LargeMovieDataset(Dataset):

    def __init__(self,
                 train=True,
                 train_split=0.8,
                 data_path="G:/web_and_text_project/data/Large_movie_dataset/aclImdb/",
                 recover_serialized=True):
        """
        Create a dataset object
        :param train: Give True if train set or False if test set
        :param train_split: The proportion of the dataset used for the train set
        :param data_path: The path to the dataset folder
        :param recover_serialized: If False: Create the dataset from the initial data:
        Load all text and labels, tokenize the text and create a dictionary of all tokens
        and finally save all in a pickle (takes some time and do this for both train and)
        test set)
        If False: Try to recover a previously serialized dataset (fast)

        """

        # The path to the large movie dataset folder
        self.data_path = data_path
        # The proportion of the dataset used for training purposes
        self.train_split = train_split
        # If we instanciating a training set, a testing set if false
        self.train = train
        # The seed used to shuffle
        self.seed = 1

        # If we have to prepare a new dataset from raw data
        if not recover_serialized:
            # Get data
            self.data_text = []
            self.data_sentiment = []
            # Different folders who contain data
            sub_dir_lst = ['test/neg', 'test/pos', 'train/neg', 'train/pos']
            idx = 0
            for sub in sub_dir_lst:
                # Get the sentiment of the folder
                sentiment = 0   # Negative sentiment
                if idx == 1 or idx == 3:
                    sentiment = 1

                # Get list of files in the folder
                sub_lst = os.listdir('{}{}'.format(self.data_path, sub))

                # Read and store all files in the list
                for itm in sub_lst:

                    f = open('{}{}/{}'.format(self.data_path, sub, itm), 'r', encoding='utf8')
                    readed = f.read()
                    # Some modifications in the data
                    readed.replace('<bt />', '')
                    readed.replace('.', ' . ')
                    readed.replace('-', ' - ')
                    readed.replace('/', ' / ')
                    self.data_text.append(readed)
                    f.close()
                    self.data_sentiment.append(sentiment)

                idx += 1

            # Shuffle the dataset using fix seed
            shuf_idx = np.arange(len(self.data_text))
            np.random.shuffle(shuf_idx)
            shuf_idx = shuf_idx.tolist()
            tmp_txt = [self.data_text[i] for i in shuf_idx]
            tmp_sent = [self.data_sentiment[i] for i in shuf_idx]
            self.data_text = tmp_txt
            self.data_sentiment = tmp_sent

            # tokenize data
            self.data_tokens = []
            for itm in self.data_text:
                self.data_tokens.append(nltk.word_tokenize(itm.lower()))

            # Get a dictionnary with all tokens
            self.dictionary = {}
            word_idx = 0
            for itm in self.data_tokens:
                for tok in itm:
                    if tok in self.dictionary.keys():
                        continue
                    else:
                        self.dictionary[tok] = word_idx
                        word_idx += 1

            # Build tensors word index (so not true one hot encoding
            self.data_one_hot = []
            for sentence in self.data_tokens:
                sn_tns = torch.zeros(len(sentence))
                for i in range(0, len(sentence)):
                    sn_tns[i] = self.dictionary[sentence[i]]
                self.data_one_hot.append(sn_tns)

            # split train and test_set
            split_idx = int(len(self.data_text) * self.train_split)
            self.data_text_train = self.data_text[0:split_idx]
            self.data_sentiment_train = self.data_sentiment[0:split_idx]
            self.data_tokens_train = self.data_tokens[0:split_idx]
            self.data_one_hot_train = self.data_one_hot[0:split_idx]

            self.data_text_test = self.data_text[split_idx:]
            self.data_sentiment_test = self.data_sentiment[split_idx:]
            self.data_tokens_test = self.data_tokens[split_idx:]
            self.data_one_hot_test = self.data_one_hot[split_idx:]

            # Store the prepared dataset to a pickle
            with open('{}/train_seri.pkl'.format(self.data_path), 'wb') as f:
                pickle.dump([self.data_text_train,
                             self.data_tokens_train,
                             self.data_sentiment_train,
                             self.dictionary,
                             self.data_one_hot_train], f)

            with open('{}/test_seri.pkl'.format(self.data_path), 'wb') as f:
                pickle.dump([self.data_text_test,
                             self.data_tokens_test,
                             self.data_sentiment_test,
                             self.dictionary,
                             self.data_one_hot_test], f)

        # Unserialize data
        if train:
            with open('{}/train_seri.pkl'.format(self.data_path), 'rb') as f:
                self.data_text, self.data_tokens, self.data_sentiments, self.dictionary, self.data_one_hot = pickle.load(f)
        else:
            with open('{}/test_seri.pkl'.format(self.data_path), 'rb') as f:
                self.data_text, self.data_tokens, self.data_sentiments, self.dictionary, self.data_one_hot = pickle.load(f)


    def __len__(self):

        return len(self.data_sentiments)

    def __getitem__(self, index):

        return self.data_one_hot[index], self.data_sentiments[index]














if __name__ == '__main__':


    dataset = LargeMovieDataset(train=True)

    print(len(dataset.data_one_hot))

    dataset.__getitem__(index=29)

    print(dataset.dictionary)
