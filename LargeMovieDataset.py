import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import sys
import pickle
import torch
from tokenizers import BertWordPieceTokenizer
import requests

class LargeMovieDataset(Dataset):

    def __init__(self,
                 train=True,
                 train_split=0.8,
                 data_path="G:/web_and_text_project/data/Large_movie_dataset/aclImdb/",
                 recover_serialized=True,
                 device='cpu'):
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
        # Device for output tensors
        self.device = device

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

            # tokenize data: note: vocabulary file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
            self.tokenizer = BertWordPieceTokenizer('models/bert_tokenizer_dic/bert_tok.txt', lowercase=True)
            self.token_seq = []
            # Store also index sequences
            self.idx_seq = []
            for itm in self.data_text:
                tok_seq = self.tokenizer.encode(itm.lower())
                self.token_seq.append(tok_seq.tokens)
                self.idx_seq.append(tok_seq.ids)

            # Get a dictionnary with all tokens
            self.dictionary = {}
            # Load bert pre trained dict
            f = open('models/bert_tokenizer_dic/bert_tok.txt', 'r', encoding='utf-8')
            raw_dic = f.read()
            raw_dic = raw_dic.split('\n')
            f.close()
            for i in range(0, len(raw_dic)):
                word = raw_dic[i]
                self.dictionary[str(word)] = i

            # Build an inverse dictionary to get tokens from index
            self.dictionary_inv = {}
            for key in self.dictionary.keys():
                idx = self.dictionary[key]
                self.dictionary_inv[str(idx)] = key

            # split train and test_set
            split_idx = int(len(self.data_text) * self.train_split)
            self.data_text_train = self.data_text[0:split_idx]
            self.data_sentiment_train = self.data_sentiment[0:split_idx]
            self.token_seq_train = self.token_seq[0:split_idx]
            self.idx_seq_train = self.idx_seq[0:split_idx]

            self.data_text_test = self.data_text[split_idx:]
            self.data_sentiment_test = self.data_sentiment[split_idx:]
            self.token_seq_test = self.token_seq[split_idx:]
            self.idx_seq_test = self.idx_seq[split_idx:]

            # Store the prepared dataset to a pickle
            with open('{}/train_seri.pkl'.format(self.data_path), 'wb') as f:
                pickle.dump([self.data_text_train,
                             self.token_seq_train,
                             self.data_sentiment_train,
                             self.dictionary,
                             self.dictionary_inv,
                             self.idx_seq_train], f)

            with open('{}/test_seri.pkl'.format(self.data_path), 'wb') as f:
                pickle.dump([self.data_text_test,
                             self.token_seq_test,
                             self.data_sentiment_test,
                             self.dictionary,
                             self.dictionary_inv,
                             self.idx_seq_test], f)

        # Unserialize data
        if train:
            with open('{}/train_seri.pkl'.format(self.data_path), 'rb') as f:
                self.data_text, self.tokens_seq, self.data_sentiments, self.dictionary, self.dictionary_inv, self.idx_seq = pickle.load(f)
        else:
            with open('{}/test_seri.pkl'.format(self.data_path), 'rb') as f:
                self.data_text, self.token_seq, self.data_sentiments, self.dictionary, self.dictionary_inv, self.idx_seq = pickle.load(f)


    def __len__(self):

        return len(self.data_sentiments)

    def __getitem__(self, index):

        # Get indexes of each words
        idx_seq = self.idx_seq[index]
        # Build one hot encoding from indexes
        one_hot = torch.zeros(len(idx_seq), len(self.dictionary.keys()))
        for i in range(0, len(idx_seq)):
            idx = int(idx_seq[i])
            one_hot[i, idx] = 1

        return one_hot.to(self.device), self.data_sentiments[index]














if __name__ == '__main__':
    """
    Example of usage
    """

    # Instanciate the dataset
    dataset = LargeMovieDataset(train=True)

    # Get a data entry
    one_hot_sentence, sentiment = dataset.__getitem__(index=29)
    one_hot_sentence = one_hot_sentence.cpu().numpy()
    print(one_hot_sentence.shape)
    # Get words index:
    word_idx = np.argmax(one_hot_sentence, axis=1)

    # Get the index of each word by an argmax in the one hot encoding
    # and get the word in the inverse dictionary
    sentence = []
    for idx in word_idx:

        sentence.append(dataset.dictionary_inv[str(idx)])

    final_text = ' '.join(sentence)
    print(final_text)


