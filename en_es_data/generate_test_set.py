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
def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    max_len = 0
    for sentence in sents:
        l = len(sentence)
        if l > max_len:
            max_len = l
    for sentence in sents:
        l = len(sentence)
        if l < max_len:
            for i in range(l,max_len):
                sentence.append(pad_token)
        sents_padded.append(sentence)

    ### END YOUR CODE

    return sents_padded

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
    data = pad_sents(data,'<UNK>')
    maxlen = 0;
    for sent in data:
        l = len(sent)
        if maxlen < l:
            maxlen = l

    print (maxlen)
    c = 0
    for i in data:
        print(len(i))
        c += 1
    print (c)

