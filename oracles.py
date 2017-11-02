import logging
import os
import gensim
import networkx as nx
import numpy as np

from gensim.models import LdaModel
from scipy.stats import entropy as kld

from lib.cnn_dmm_utils import chunk_list
from train_topic_models import TrainTopicModel

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = 'models/'

class oracle(object):
    def __init__(self, dictionary, corpus, env, passes=4, threshold=.9, num_topics=100, window=2, model='lda', model_name='lda'):
        self.dictionary = dictionary
        self.corpus = corpus
        self.env = env
        self.threshold = threshold
        self.window = window
        self.model = model
        self.passes = passes 
        self.model_name = model_name
        self.num_topics = num_topics
        self.models = []
        self.train_models()

    def train_models(self):
        corpus_len = len(self.corpus)
        logger.info('No of docs in corpus {}'.format(corpus_len))
        model = TrainTopicModel(self.model, articles=[], corpus=self.corpus,
                                dictionary=self.dictionary, preprocess=False, num_topics=self.num_topics)
        model = model.train_model(passes=self.passes, saved_model=self.model_name)
        self.models.append(model)
        model.save(os.path.join(MODELS_DIR, self.model_name))

    def get_active_topics(self, docs):
        active_topics = set()
        model = self.models[0]
        if not docs:
            return active_topics
        for doc in docs:
            topics = model.get_document_topics(doc[1], minimum_probability=.15)
            for topic in topics:
                active_topics |= set([topic[0]])
        return active_topics

    def imp_documents(self, docs, round_no):
        model = self.models[0]
        t_topics = [(doc[0], model.get_document_topics(doc[1], minimum_probability=.15)) for doc in docs]
        if round_no <= self.window or np.random.uniform() > self.threshold:
            logger.info("Resample docs")
            return [docs[i] for i in range(len(docs))]
        self.env.current_idx = docs[0][0]
        prev_topics = set()
        for day in range(1,self.window+1):
            prev_topics |= self.get_active_topics(self.env.prev(n=day))
        logger.info("Active Topics in the window {}".format(len(prev_topics)))
        docs_chosen = []
        for t_topic in t_topics:
            topics = set([topic[0] for topic in t_topic[1]])
            logger.info("Topics in set {}".format(len(topics)))
            if bool(topics & prev_topics):
                docs_chosen.append(t_topic[0])
        return docs_chosen
