import os
import logging
from gensim.models import LdaModel

from preprocess_cnn_dmm_data import tokenize_articles, lemmatize_articles
from preprocess_cnn_dmm_data import construct_dict, construct_corpus

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = os.path.abspath('models')

class TrainTopicModel(object):
    def __init__(self, model_name, articles, corpus_path=None, corpus_name='', 
                dictionary_path=None, dict_name='', num_topics=100, preprocess=True):
        self.model_name = model_name
        self.articles = articles
        self.corpus_path = corpus_path
        self.corpus_name = corpus_name
        self.dictionary_path = dictionary_path
        self.dict_name = dict_name
        self.num_topics = num_topics
        self.preprocess = preprocess
        if self.preprocess:
            self.preprocess_articles()
            self.create_dictionary(create_new=True)
            self.create_corpus(create_new=True)

    def preprocess_articles(self):
        logger.info('tokenizing articles')
        self.articles = tokenize_articles(self.articles)
    
        logger.info('removing stopwords')
        self.articles = remove_stop_words(self.articles)

        logger.info('lemmatizing articles')
        self.articles = lemmatize_articles(self.articles)

    def create_dictionary(self, create_new=False):
        if self.dict_name:
            self.dictionary = construct_dict(self.articles, self.dictionary_path, self.dict_name, create_new)
        else:
            self.dictionary = construct_dict(self.articles, self.dictionary_path, create_new)

    def create_corpus(self, create_new=False):
        if self.corpus_name:
            self.corpus = construct_corpus(self.articles, self.dictionary,
                                           self.corpus_path, self.corpus_name, create_new)
        else:
            self.corpus = construct_corpus(self.articles, self.dictionary, self.corpus_path, create_new)

    def run_training(self):
        if self.model_name == 'lda':
            self.create_dictionary()
            self.create_corpus()
            self.model = LdaModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics)
            logger.info('Completed training')
            logger.info('Trained model saved at {}'.format(MODELS_DIR))
            self.model.save(os.path.join(MODELS_DIR, 'lda'))

    def train_model(self, create_new=False):
        self.model_name = self.model_name.lower()
        if create_new:
            self.run_training()
        if self.model_name == 'lda':
            try:
                self.model = LdaModel.load(os.path.join(MODELS_DIR, 'lda'))
            except:
                self.run_training()
        return self.model 
