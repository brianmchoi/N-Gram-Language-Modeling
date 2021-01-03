#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Zimeng Qiu Sep 2019

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse
from utils import *
import math 
from collections import Counter
import copy


class LanguageModel(object):
    """
    Base class for all language models
    """
    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        # write your initialize code below
        self.corpus = corpus
        self.ngram = ngram
        self.min_freq = min_freq
        self.uniform = uniform
        self.unk_list = []
        self.uni_dist = {}
        self.uni_model = {}
        self.bi_model = {}
        self.bi_list = {}
        self.tri_model = {}
        self.tri_list = {}
        self.build()
        #raise NotImplemented

    def unk_replace(self):
        for i in range(len(self.corpus)):
            for j in range(len(self.corpus[i])):
                if self.corpus[i][j] in self.unk_list:
                    self.corpus[i][j] = "UNK"

    def build(self):
        """
        Build LM from text corpus
        """
        # word_dict tallies each word in corpus
        word_dict = {}
        for sent in self.corpus:
            for word in sent:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        #print(len(word_dict))
        # replace words that appear less than the min_freq allowed
        for word in word_dict:
            if word_dict[word] < self.min_freq:
                self.unk_list.append(word)
        # after identifying words to replace with unk, run helper
        self.unk_replace()

        # build dictionaries based on n-gram
        if self.ngram == 1 and self.uniform == True:
            self.uni_dist = {}
            for sent in self.corpus:
                for word in sent:
                    self.uni_dist[word] = 1

        if self.ngram == 1 and self.uniform == False:
            #self.uni_model = {}
            for sent in self.corpus:
                for word in sent:
                    if word in self.uni_model.keys():
                        self.uni_model[word] += 1
                    else:
                        self.uni_model[word] = 1
        
        if self.ngram == 2:
            self.bi_model = {}
            for sent in self.corpus:
                for i in range(len(sent)-1):
                    bigram = sent[i] + " " + sent[i+1]
                    if bigram in self.bi_model:
                        self.bi_model[bigram] += 1
                    else:
                        self.bi_model[bigram] = 1
            self.bi_list = copy.deepcopy(self.bi_model)
            for sent in self.corpus:
                for i in range(len(sent)):
                    unigram = sent[i]
                    if unigram in self.bi_model:
                        self.bi_model[unigram] += 1
                    else:
                        self.bi_model[unigram] = 1
            
        if self.ngram == 3:
            self.tri_model = {}
            for sent in self.corpus:
                for i in range(len(sent)-2):
                    trigram = sent[i] + " " + sent[i+1] + " " + sent[i+2]
                    if trigram in self.tri_model:
                        self.tri_model[trigram] += 1
                    else:
                        self.tri_model[trigram] = 1
            self.tri_list = copy.deepcopy(self.tri_model)
            for sent in self.corpus:
                for i in range(len(sent)-1):
                    bigram = sent[i] + " " + sent[i+1]
                    if bigram in self.tri_model:
                        self.tri_model[bigram] += 1
                    else:
                        self.tri_model[bigram] = 1

        #raise NotImplemented

    def most_common_words(self, k):
        """
        Return the top-k most frequent n-grams and their frequencies in sorted order.
        For uniform models, the frequency should be "1" for each token.

        Your return should be sorted in descending order of frequency.
        Sort according to ascending alphabet order when multiple words have same frequency.
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        if self.ngram == 1 and self.uniform == True:
          n_dict = self.uni_dist 
        if self.ngram == 1 and self.uniform == False:
          n_dict = self.uni_model
        if self.ngram == 2:
          n_dict = self.bi_list
        if self.ngram == 3:
          n_dict = self.tri_list

        sorted_dict = dict(sorted(n_dict.items(), key=lambda x: x[0].lower()) )

        cnt = Counter(sorted_dict)
        common_list = cnt.most_common(k)

        return common_list

        #raise NotImplemented


def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    #initialize perplexity, test file length, vocab list
    ppl, test_len = 0, 0
    word_list = []

    for sent in models[0].corpus:
        for word in sent:
            word_list.append(word)
    V = len(set(word_list))

    for policy in data:
        test_len += len(policy)
    #replace unseen words in test to "UNK"
    vocabulary = set(word_list)
    for policy in data:
        for word in policy:
            if word not in vocabulary:
                index = policy.index(word)
                policy[index] = "UNK"

    #calculate probabilty of each word using linear interpolation
    for policy in data:
        for word in policy:
            word_prob = 0
            for model in models:
                #uniform or unigram case
                if model.ngram == 1 and model.uniform == True:
                    word_prob += (coefs[0] * (2/(V)))
                #unigram case
                elif model.ngram == 1 and model.uniform == False:
                    C_i = model.uni_model[word] + 1
                    word_prob += (coefs[1] * ((C_i + 1) / (test_len + V)))
                #bigram case
                elif model.ngram == 2:
                    idx = policy.index(word)
                    if idx > 0:
                        bigram = policy[idx-1] + " " + policy[idx]
                        unigram = policy[idx-1]
                        C_i = model.bi_model.get(bigram, 0)
                        C_prev = model.bi_model.get(unigram, 0)
                        #get counts and then calculate prob
                        word_prob += (coefs[2] * ((C_i + 1) / (C_prev + V))) ** (1/model.ngram)
                    else:
                        word_prob += 0
                #trigram case
                elif model.ngram == 3:
                    idx = policy.index(word)
                    if idx > 1:
                        trigram = policy[idx-2] + " " + policy[idx-1] + " " + policy[idx]
                        bigram = policy[idx-2] + " " + policy[idx-1]
                        C_i = model.tri_model.get(trigram, 0)
                        C_prev = model.tri_model.get(bigram, 0)
                        #get counts and calc prob
                        word_prob += (coefs[3] * ((C_i + 1) / (C_prev + V))) ** (1/model.ngram)
                    else:
                        word_prob += 0
            #if coef of model > 0, then increment our perplexity value
            if word_prob > 0:
                ppl += math.log2(word_prob)
            #if not, move on to next
            else:
                ppl += 0
    #use perplexity formula given in class lecture slides
    ppl = math.pow(2, (-1*ppl) / (test_len))                    
    return ppl

    #raise NotImplemented


# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    return parser.parse_args()


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    print("Training Uniform...")
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)


    print("Training Unigram...")
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    print("Length of n-gram for unigrams:")
    print(len(uniform.uni_dist))


    print("Training Bigram...")
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    print("Length of n-gram for bigrams:")
    print(len(bigram.bi_list))


    print("Training Trigram...")
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)
    print("Length of n-gram for trigrams:")
    print(len(trigram.tri_list))
    print("Training done!")

    # calculate perplexity on test file
    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))

