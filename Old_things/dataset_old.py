import torch
import pandas
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import CamembertTokenizer
import pickle
#from elmoformanylangs import Embedder


class AC_Dataset(Dataset):

    def __init__(self,
                 data_path='C:/Users/franc/Documents/Unif/BigData Project/covid-mood-analysis/CamemBERT/Train_data/allocine_dataset.pickle',
                 max_len=300,       # Max number of characters in inputs
                 train_split=0.8,   # Proportion of the dataset for training purposes
                 train=False,       # Test set if false
                 reduce=True,       # Reduce the size for testing purposes
                 embed='no',      # Embedding model
                 ):

        self.max_len = max_len
        self.train_split = train_split
        self.data_path = data_path
        self.train = train
        self.reduce = reduce
        self.reduce_size = 1000
        self.embed=embed

        # An object to perform the embedding
        self.embedder = None

    def import_allocine_data(self):

        # Unpickle data
        with open(self.data_path, 'rb') as reader:
            data = pickle.load(reader)

        # Load all data
        print('Get data array...')
        train_rev = data["train_set"]['review'].to_numpy()
        val_rev = data["val_set"]['review'].to_numpy()
        test_rev = data["test_set"]['review'].to_numpy()
        # import labels
        train_labels = data["train_set"]['polarity'].to_numpy()
        val_labels = data["val_set"]['polarity'].to_numpy()
        test_labels = data["test_set"]['polarity'].to_numpy()
        class_names = data['class_names']
        print('...Done')
        del data
        # Concat data
        reviews = np.concatenate([train_rev, val_rev, test_rev])
        # Concat labels
        sentiments = np.concatenate([train_labels, val_labels, test_labels])
        # To list
        reviews = reviews.tolist()

        # If reduce (doesn't load all the dataset
        if self.reduce is not None:
            print('WARNING: reduced dataset: for deboging purposes only')
            reviews = reviews[0:self.reduce_size]
            sentiments = sentiments[0:self.reduce_size]

        # Split train and test parts
        split_border = int(len(sentiments) * self.train_split)
        if self.train:
            reviews = reviews[0:split_border]
            sentiments = sentiments[0:split_border]
        else:
            reviews = reviews[split_border:]
            sentiments = sentiments[split_border:]

        # Embed the text
        if self.embed == 'elmo':
            print('Initializing Elmo embedding...')
            self.embedder = Embedder('EmbedModels/elmo/', batch_size=64)
            print('... Done')

        # Perform embedding
        #embedded = self.embedder.sents2elmo(reviews)

        # Camembert tokenization
        self.embedder = CamembertTokenizer.from_pretrained(
            'camembert-base',
            do_lower_case=True
        )
        encoded_batch = self.embedder.batch_encode_plus(reviews,
                                                         add_special_tokens=True,
                                                         max_length=self.max_len,
                                                         padding=True,
                                                         truncation=True,
                                                         return_attention_mask=True,
                                                         return_tensors='pt')

        #for i in range(0, 10):
            #print(reviews[i])
            #print(encoded_batch['attention_mask'][i])




if __name__ == '__main__':

    dataset = AC_Dataset()

    dataset.import_allocine_data()
