import logging
import os
import gensim
import networkx as nx

from gensim.models import LdaModel
from lib.cnn_dmm_utils import chunk_list
from train_topic_models import TrainTopicModel

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = 'models/'

class TopicChains(object):
    def __init__(self, dictionary, corpus, env, chunk_size=4000, model_name='lda'):
        self.dictionary = dictionary
        self.corpus = corpus
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.models = []

    def articles_from_env(self):
        pass

    def load_models(self):
        try:
            for model_no in range(len(self.corpus)/self.chunk_size):
                mname = 'lda' + str(idx)
                model = LdaModel.load(os.path.join(MODELS_DIR, mname))
                self.models.append(model)
            return
        except:
            self.models = []
                
    def train_models(self):
        self.load_models()

        if self.models:
            return

        corpus_len = len(self.corpus)
        logger.info('No of docs in corpus {}'.format(corpus_len))
        no_models = corpus_len/self.chunk_size
        logger.info('No of LDA models {}'.format(no_models))
        for idx, corpus in enumerate(chunk_list(self.corpus, self.chunk_size)):
            logger.info('Corpus length {}'.format(len(corpus)))
            model = TrainTopicModel(self.model_name, articles=[], corpus=corpus,
                                    dictionary=self.dictionary, preprocess=False, num_topics=20)
            mname = 'lda' + str(idx)
            model = model.train_model(saved_model=mname, passes=20)
            self.models.append(model)
            model.save(os.path.join(MODELS_DIR, mname))
            
    def construct_topic_chains(self):
        self.train_models()

    def imp_documents(self, input_docs):
        pass
