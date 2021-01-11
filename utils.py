import copy
import json
import math
import pickle
import random
import re
from collections import Counter

import nltk
import torch
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

torch.autograd.set_detect_anomaly(True)

# https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699/21
def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        else:
            raise ValueError('Padding is supported for upto 3D tensors at max.')
    return padded_tensor


def ints_to_tensor(ints):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], int):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return pad_tensors(ints)
        if isinstance(ints[0], list):
            return ints_to_tensor([ints_to_tensor(inti) for inti in ints])


def get_mask(node_num, max_edu_dist):
    batch_size, max_num=node_num.size(0), node_num.max()
    mask=torch.arange(max_num).unsqueeze(0).cuda()<node_num.unsqueeze(1)
    mask=mask.unsqueeze(1).expand(batch_size, max_num, max_num)
    mask=mask&mask.transpose(1,2)
    mask = torch.tril(mask, -1)
    if max_num > max_edu_dist:
        mask = torch.triu(mask, max_edu_dist - max_num)
    return mask


def compute_loss(link_scores, label_scores, graphs, mask, p=False, negative=False):
    link_scores[~mask]=-1e9
    label_mask=(graphs!=0)&mask
    tmp_mask=(graphs.sum(-1)==0)&mask[:,:,0]
    link_mask=label_mask.clone()
    link_mask[:,:,0]=tmp_mask
    link_scores=torch.nn.functional.softmax(link_scores, dim=-1)
    link_loss=-torch.log(link_scores[link_mask])
    vocab_size=label_scores.size(-1)
    label_loss=torch.nn.functional.cross_entropy(label_scores[label_mask].reshape(-1, vocab_size), graphs[label_mask].reshape(-1), reduction='none')
    if negative:
        negative_mask=(graphs==0)&mask
        negative_loss=torch.nn.functional.cross_entropy(label_scores[negative_mask].reshape(-1, vocab_size), graphs[negative_mask].reshape(-1),reduction='mean')
        return link_loss, label_loss, negative_loss
    if p:
        return link_loss, label_loss, torch.nn.functional.softmax(label_scores[label_mask],dim=-1)[torch.arange(label_scores[label_mask].size(0)),graphs[mask]]
    return link_loss, label_loss


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def record_eval_result(eval_matrix, predicted_result):
    for k, v in eval_matrix.items():
        if v is None:
            if isinstance(predicted_result[k], dict):
                eval_matrix[k] = [predicted_result[k]]
            else:
                eval_matrix[k] = predicted_result[k]
        elif isinstance(v, list):
            eval_matrix[k] += [predicted_result[k]]
        else:
            eval_matrix[k] = np.append(eval_matrix[k], predicted_result[k])


def get_error_statics(eval_matrix):
    # error type: 0 link error, 1 label error
    errors_0 = []
    errors_1 = []
    errors_dist_0 = [0] * 20
    errors_dist_1 = [0] * 20
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        for h_p, h_r in hypothesis.items():
            h_x = h_p[0]
            h_y = h_p[1]
            for r_p, r_r in reference.items():
                r_x = r_p[0]
                r_y = r_p[1]
                if h_y == r_y and r_x < r_y:
                    if h_x == r_x and h_r != r_r:
                        errors_1.append((h_r, r_r))
                        errors_dist_1[h_y - h_x] += 1
                    elif h_x != r_x:
                        errors_0.append((h_r, r_r))
                        errors_dist_0[h_y - h_x] += 1
    return sorted(Counter(errors_0).items(), key=lambda x: x[1], reverse=True), sorted(Counter(errors_1).items(),
                                                                                       key=lambda x: x[1],
                                                                                       reverse=True), errors_dist_0, errors_dist_1

def survey(eval_matrix, id2types):
    survey_dict={}
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        for pair in reference:
            label=reference[pair]
            if label not in survey_dict:
                survey_dict[label]=[0, 0, 1]
            else:
                survey_dict[label][2]+=1
            if pair in hypothesis:
                survey_dict[label][0]+=1
                if hypothesis[pair] == reference[pair]:
                    survey_dict[label][1]+=1
    for k, v in survey_dict.items():
        print(id2types[k], v[0], v[1], v[2], v[0]*1.0/v[2], v[1]*1.0/v[2])


def test_F1(eval_matrix):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        cnt_golden += len(reference)
        for pair in hypothesis:
            if pair[0] != -1:
                cnt_pred += 1
                if pair in reference:
                    cnt_cor_bi += 1
                    if hypothesis[pair] == reference[pair]:
                        cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi


def accuray_dist(eval_matrix):
    dist_sum=np.zeros(15)
    dist_yes=np.zeros(15)
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        for pair in reference:
            dist_sum[pair[1]]+=1
            if pair in hypothesis and hypothesis[pair] == reference[pair]:
                dist_yes[pair[1]]+=1
    print(dist_yes/dist_sum)
    print(dist_sum)
    print(dist_yes.sum()/dist_sum.sum())


def tsinghua_F1(eval_matrix):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference, edu_num in zip(eval_matrix['hypothesis'], eval_matrix['reference'],
                                              eval_matrix['edu_num']):
        cnt = [0] * edu_num
        for r in reference:
            cnt[r[1]] += 1
        for i in range(edu_num):
            if cnt[i] == 0:
                cnt_golden += 1
        cnt_pred += 1
        if cnt[0] == 0:
            cnt_cor_bi += 1
            cnt_cor_multi += 1
        cnt_golden += len(reference)
        cnt_pred += len(hypothesis)
        for pair in hypothesis:
            if pair in reference:
                cnt_cor_bi += 1
                if hypothesis[pair] == reference[pair]:
                    cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi


class GloveTokenizer:
    def __init__(self, args):
        self.args = args
        self.glove_vocab = self.load_glove_embedding()
        self.fdist = self.load_corpus()
        self.word2idx, self.emb = self.corpus_vocab()
        self.pad_token_id = 0
        self.glove_vocab = None

    def load_glove_embedding(self):
        glove_vocab = {}
        with open(self.args.glove_vocab_path, 'r')as file:
            for line in file.readlines():
                line = line.split()
                glove_vocab[line[0]] = np.array(line[1:]).astype(np.float)
        return glove_vocab

    def encode(self, text, special_token=True):
        if special_token:
            return [self.word2idx['CLS']] + [self.word2idx[word] if word in self.word2idx else self.word2idx['UNK'] for
                                             word in
                                             self.tokenize(text)]
        else:
            return [self.word2idx[word] if word in self.word2idx else self.word2idx['UNK'] for word in
                    self.tokenize(text)]

    @staticmethod
    def convert_number_to_special_token(tokens):
        # number to special token
        for i, token in enumerate(tokens):
            if re.match("\d+", token):
                tokens[i] = "[num]"
        return tokens

    @staticmethod
    def tokenize(text):
        return GloveTokenizer.convert_number_to_special_token(nltk.word_tokenize(text.lower()))

    def load_corpus(self):
        corpus_words = []
        for corpus_file in (self.args.train_file, self.args.eval_file, self.args.test_file):
            with open(corpus_file, 'r')as file:
                dataset = json.load(file)
                for data in dataset:
                    for edu in data['edus']:
                        corpus_words += self.tokenize(edu['text'])
        fdist = nltk.FreqDist(corpus_words)
        fdist = sorted(fdist.items(), reverse=True, key=lambda x: x[1])
        vocab = []
        for i, word in enumerate(fdist):
            word = word[0]
            if i < self.args.max_vocab_size or word in self.glove_vocab:
                vocab.append(word)
        return vocab

    def corpus_vocab(self):
        word2idx = {'PAD': 0, 'UNK': 1, 'CLS': 2, 'EOS': 3}
        define_num = len(word2idx)
        emb = [np.zeros(self.args.glove_embedding_size)] * define_num
        for idx, word in enumerate(self.fdist):
            word2idx[word] = idx + define_num
            if word in self.glove_vocab:
                emb.append(self.glove_vocab[word])
            else:
                emb.append(np.zeros(self.args.glove_embedding_size))
        print('corpus size : {}'.format(idx + define_num + 1))
        return word2idx, emb
