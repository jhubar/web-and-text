import pandas as pd
import numpy as np


elements = []
f = open('models/bert_tokenizer_dic/bert_tok.txt', 'r', encoding='utf-8')

text = f.read()

dic = text.split('\n')

for itm in dic:
    print(itm)