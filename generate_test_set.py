import numpy as np

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

if __name__ == '__main__':
    data = read_corpus('train.en','tgt')

    maxlen = 0;
    for sent in data:
    	l = len(sent)
    	if maxlen < l:
    		maxlen = l

	print (maxlen)
    for i in range(10):
    	print(data[i],"\n")
    	print(len(data[i]))


