from __future__ import absolute_import
from __future__ import division
#把下一个新版本的特性导入到当前版本
from __future__ import print_function

import os
import nltk
import numpy as np
import pickle
import random
import json

padToken,goToken,eosToken,unknownToken = 0,1,2,3

class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def loadDataset():
    with open("voc_list.txt") as g:
        voc = g.readlines()
    vocab = []
    for word in voc[0:60001]:
        vocab.append(word.strip().split("\t")[0])

    # 获得所有的one2one对
    f = open("kp20k_training.json")
    samples = []
    for line in f.readlines():
        t = json.loads(line)
        for j in range(len(t["keyword"].split(";"))):
            sample = []
            sample.append(t["abstract"])
            sample.append(t["keyword"].split(";")[j])
            samples.append(sample)

    # 标识符及其索引
    symbols = {0: "<PAD>", 1: "<UNK>", 2: "<GO>", 3: "<EOS>"}

    int_to_vocab = {}

    # 得到词表的索引表int_to_vocab
    for index_no, word in enumerate(vocab):
        int_to_vocab[index_no] = word

    # 将symbols的键值对更新到int_to_vocab中
    int_to_vocab.update(symbols)

    vocab_to_int = {word: index_no for index_no, word in int_to_vocab.items()}
    # 对于源文本的每一行（即每一个句子）得到索引结构
    data = []
    for sample in samples:
        data_ = []
        datasource = []
        datatarget = []

        for sword in sample[0].split(" "):
            if sword.lower() in vocab_to_int:
                datasource.append(vocab_to_int[sword.lower()])
        data_.append(datasource)
        for tword in sample[1].split(" "):
            if tword.lower() in vocab_to_int:
                datatarget.append(vocab_to_int[tword.lower()])
        data_.append(datatarget)

        data.append(data_)
    print(len(data))
    return int_to_vocab,vocab_to_int,data
def createBatch(samples):
    '''
    :param samples: 一个batch的数据
    :return: 可直接传入feeddict的一个batch的数据格式
    '''
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) for sample in samples]


    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        #将source进行反序
        source= list(reversed(sample[0]))
        pad = [padToken] * (max_source_length-len(source))
        batch.encoder_inputs.append(pad+source)

        target = sample[1]
        pad = [padToken] * (max_target_length-len(target))
        batch.decoder_targets.append(pad+target)

    return batch

def getBatches(data,batch_size):

    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():

        for i in range(0,data_len,batch_size):

            yield data[i:min(i+batch_size,data_len)]

    for sample in genNextSamples():
        batch = createBatch(sample)
        batches.append(batch)
    return batches


