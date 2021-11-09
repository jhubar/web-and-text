import torch
import pandas
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import pickle
import os

class TrainSetBuilder():

    def __init__(self,
                 data_path="D:/web_and_text_project/data/Large_movie_dataset/aclImdb/",
                 load_pickle=True,
                 num_workers=0,
                 batch_size=10,
                 pin_memory=False):
        """
        This class manage the building of the LargeMovieDataset building
        and use.
        :param data_path: the path to the aclImdb folder of the dataset
        :param load_pickle: If it's the first run: select false to build
        a serialized dataset (slow)
        If serialized dataset already build: turn on True for fast opening
        :param num_workers: Num of workers for the data data loaders
        :param batch_size: The size of batches produces by data loaders

        Use "inport_cine_data()" to prepare the dataset before use
        Use "get_dataloaders()" to obtain train and test dataloaders

        Batch outputs:
            - batch[0] = index sequence (tensor)
            - batch[1] = attention mask (contain ones for each words and
                    zeros in front of padding tokens
            - batch[2] = sentiments (tensor)
        """

        # Hyper parameters
        self.max_len = 500      # Maximum number of words in a sequence
        self.train_split = 0.8  # Propotion of the dataset for training set
        self.batch_size = batch_size    # Number of elements in a batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Training dataset
        self.train_dataset = None
        # Testing dataset
        self.test_dataset = None
        # Classes
        self.class_labels = ['Negative', 'Positive']

        # Data handlers
        self.train_handler = None
        self.test_handler = None

        # Load the tokenizer
        self.tokenizer = BertTokenizer(vocab_file='models/bert_tokenizer_dic/bert_tok.txt',
                                       do_lower_case=True)

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

        # Pahts
        self.data_path = data_path
        # If want to load from scratch or not
        self.load_pickl = load_pickle

    def import_cine_data(self, reduce=None):

        if not self.load_pickl:
            # Load raw dataset
            data_text = []
            data_sentiment = []
            sub_dir_lst = ['test/neg', 'test/pos', 'train/neg', 'train/pos']

            idx = 0
            for sub in sub_dir_lst:
                # Get the sentiment of the folder
                sentiment = 0  # Negative sentiment
                if idx == 1 or idx == 3:
                    sentiment = 1

                # Get list of files in the folder
                sub_lst = os.listdir('{}/{}'.format(self.data_path, sub))

                # Read and store all files in the list
                for itm in sub_lst:
                    f = open('{}{}/{}'.format(self.data_path, sub, itm), 'r', encoding='utf8')
                    readed = f.read()
                    data_text.append(readed)
                    f.close()
                    data_sentiment.append(sentiment)
                idx += 1

            # Shuffle the dataset using fix seed
            shuf_idx = np.arange(len(data_text))
            np.random.shuffle(shuf_idx)
            shuf_idx = shuf_idx.tolist()
            tmp_txt = [data_text[i] for i in shuf_idx]
            tmp_sent = [data_sentiment[i] for i in shuf_idx]
            reviews = tmp_txt
            sentiments = tmp_sent

            # Encode the batch of data
            print('Data tokenization...')
            encoded_batch = self.tokenizer.batch_encode_plus(reviews,
                                                             add_special_tokens=True,
                                                             max_length=self.max_len,
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True,
                                                             return_tensors='pt')

            # Serialize the dataset
            with open('{}/serialized_dataset.pkl'.format(self.data_path), 'wb') as f:
                pickle.dump([data_text, encoded_batch, data_sentiment], f)




        # Load serialized dataset
        with open('{}/serialized_dataset.pkl'.format(self.data_path), 'rb') as reader:
            reviews, encoded_batch, sentiments = pickle.load(reader)


        # If reduce (doesn't load all the dataset
        if reduce is not None:
            print('WARNING: reduced dataset: for deboging purposes only')
            reviews = reviews[0:reduce]
            sentiments = sentiments[0:reduce]


        print('... Done')

        # Get the spliting index
        split_border = int(len(sentiments)*self.train_split)
        # Get a tensor for sentiments
        sentiments = torch.tensor(sentiments)
        # Now encode datasets tensors
        print('Tensors encoding...')
        self.train_dataset = TensorDataset(
            encoded_batch['input_ids'][:split_border],
            encoded_batch['attention_mask'][:split_border],
            sentiments[:split_border])
        self.test_dataset = TensorDataset(
            encoded_batch['input_ids'][split_border:],
            encoded_batch['attention_mask'][split_border:],
            sentiments[split_border:])
        print('... Done')

        # Get data handler
        print('Data handler encoding...')
        self.train_handler = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

        self.test_handler = DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory)
        print('... Done')
        print('End of dataset encoding.')

    def get_data_loader(self):

        return self.train_handler, self.test_handler

    def get_tokenizer(self):

        return self.tokenizer


# Testing
if __name__ == '__main__':

    builder = TrainSetBuilder()
    builder.import_cine_data()

