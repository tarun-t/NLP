import logging
import os
import gensim
import networkx as nx
import numpy as np

from gensim.models import LdaModel
from scipy.stats import entropy as kld
from sklearn.metrics.pairwise import cosine_similarity as cs

from lib.cnn_dmm_utils import chunk_list
from train_topic_models import TrainTopicModel
from doc2vec import doc2vec

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


class d2v_oracle(object):
    def __init__(self, env, threshold=.2, window=2, model_name='doc2vec.d2v', d2v_window=10, size=100, epochs=10):
        self.model_name = model_name
        self.env = env
        self.threshold = threshold
        self.window = window
        self.d2v_window = d2v_window
        self.size = size
        self.epochs = epochs
        self.model = doc2vec(model_name=self.model_name, window=self.d2v_window, size=self.size, epochs=self.epochs).model

    def get_context(self, docs):
        return [self.model.infer_vector(doc) for doc in docs]

    def imp_documents(self, docs, round_no):
        if round_no <= self.window or np.random.uniform() > .9:
            logger.info("Resample docs")
            return [docs[i] for i in range(len(docs))]
        self.env.current_idx = docs[0][0]
        context = []
        for day in range(1,self.window+1):
            docs = [doc[1] for doc in self.env.prev(n=day)]
            context.extend(self.get_context(docs))
        docs_chosen = []
        t_docs = [(doc[0], self.model.infer_vector(doc[1])) for doc in docs]
        for t_doc in t_docs:
            for c in context:
                if cs([t_doc[1], c])[0][1] >= self.threshold:
                    docs_chosen.append(t_doc[0])
                    break
        return docs_chosen 
