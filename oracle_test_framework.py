import os
import logging
import gensim

from preprocess_cnn_dmm_data import construct_dict, construct_corpus
from preprocess_cnn_dmm_data import preprocess_articles, fetch_articles_from_data
from preprocess_cnn_dmm_data import save_dict, save_corpus
from oracles import TopicChains
from environments import ConstDocs

DICT_NAME = os.path.abspath('models/cnn_dict.dict')
CORPUS_NAME = os.path.abspath('models/cnn_corpus.mm')
TRAIN_DIR_PATH = os.path.abspath('train_data')

try:
    dictionary = gensim.corpora.Dictionary.load(DICT_NAME)
except Exception as e :
    articles = fetch_articles_from_data(TRAIN_DIR_PATH)
    articles = preprocess_articles(articles)
    dictionary = construct_dict(articles, dictionary_path=DICT_NAME)
    save_dict(dictionary)
try:
    corpus = gensim.corpora.mmcorpus.MmCorpus(CORPUS_NAME)
except:
    corpus = construct_corpus(articles, dictionary, corpus_path=CORPUS_NAME)
    save_corpus(corpus)

env = ConstDocs(corpus)
oracle = TopicChains(dictionary, corpus, env)
oracle.train_models()
