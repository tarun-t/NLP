import os
import logging
from gensim.models import LdaModel

from preprocess_cnn_dmm_data import tokenize_articles, lemmatize_articles
from preprocess_cnn_dmm_data import construct_dict, construct_corpus
from preprocess_cnn_dmm_data import read_stop_words, remove_stop_words, add_phrases

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = os.path.abspath('models')

class TrainTopicModel(object):
    def __init__(self, model_name, articles, corpus=None, corpus_path='models/cnn_corpus.mm',
                 dictionary=None, preprocess=True, dictionary_path='models/cnn_dict.dict', num_topics=100):
        self.model_name = model_name
        self.articles = articles
        self.corpus_path = corpus_path
        self.dictionary_path = dictionary_path
        self.num_topics = num_topics
        self.preprocess = preprocess
        self.corpus = corpus
        self.dictionary = dictionary
        if self.preprocess:
            self.preprocess_articles()
        if not self.dictionary:
            self.create_dictionary()
        if not self.corpus:
            self.create_corpus()

    def preprocess_articles(self):
        self.articles = tokenize_articles(self.articles)
        stop_words = read_stop_words()
        self.articles = remove_stop_words(self.articles, stop_words)
        self.articles = lemmatize_articles(self.articles)
        self.articles = add_phrases(self.articles)

    def create_dictionary(self):
        self.dictionary = construct_dict(self.articles, dictionary_path=self.dictionary_path)

    def create_corpus(self, create_new=False):
        self.corpus = construct_corpus(self.articles, self.dictionary, corpus_path=self.corpus_path)

    def train_model(self, passes=4, saved_model='lda'):
        self.model_name = self.model_name.lower()
        if self.model_name == 'lda':
            try:
                self.model = LdaModel.load(os.path.join(MODELS_DIR, saved_model))
            except:
                self.model = LdaModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=passes, update_every=0)
        return self.model 
