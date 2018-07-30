import logging
import os
import gensim
import networkx as nx
import numpy as np

from gensim.models import LdaModel
from scipy.stats import entropy as kld
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as cs
from collections import Counter

from lib.cnn_dmm_utils import chunk_list
from train_topic_models import TrainTopicModel
from doc2vec import doc2vec

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = 'models/'

class d2v_oracle(object):
    def __init__(self, env, docs_path, threshold=.2, window=2, d2v_window=10, size=100, epochs=10):
        self.env = env
        self.docs_path = docs_path
        self.threshold = threshold
        self.window = window
        self.d2v_window = d2v_window
        self.size = size
        self.epochs = epochs
        self.model = doc2vec(docs_path=self.docs_path, window=self.d2v_window, size=self.size, epochs=self.epochs).model

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
        return docs_chosen, 0


class doc_cluster_oracle(d2v_oracle):
    def __init__(self, env, docs_path, threshold=0, window=5, 
                d2v_window=10, size=100, epochs=10, clusters=100):
        d2v_oracle.__init__(self, env, docs_path, threshold, window, d2v_window, size, epochs)
        self.clusters = clusters
        self.docs_path = docs_path
        self.prev_cluster = None
        self.cluster()

    def cluster(self):
        logger.info("Clustering Docs")
        kmeans = KMeans(n_clusters=self.clusters, n_jobs=-1)
        self.kmeans = kmeans.fit(self.model.docvecs)
        logger.info("Clustering finished")

    def map_docs(self, docs):
        map ={}
        for doc in docs:
            map[doc[0]] = self.kmeans.predict(self.get_context([doc[1]]))[0]
        return map

    def get_cluster_from_trend(self):
        self.get_trending_cluster(self.env)
        active = self.mapping.values()
        max = None
        count = 0
        for act in active:
            if not max or self.trends[act] > count:
                try:
                    max = act
                    count = self.trends[act]
                except KeyError:
                    pass
        return max
        
    def get_trending_cluster(self, env):
        active = self.mapping.values()
        docs = []
        for day in range(1, self.window+1):
            docs.extend([doc for doc in env.prev(n=day)])
        mapping = self.map_docs(docs)
        self.trends = Counter(mapping.values())
        return self.trends.most_common(1)[0]
        
    def imp_documents(self, env, docs, round_no):
        self.env.current_idx = env.current_idx
        self.mapping = self.map_docs(docs)
        if round_no <= self.window:
            logger.info("pick a random cluster")
            cluster = self.mapping[docs[0][0]]
            self.prev_cluster = cluster
        elif np.random.uniform() < self.threshold and self.prev_cluster in self.mapping.values():
            cluster = self.prev_cluster
        else:
            cluster = self.get_cluster_from_trend()
            self.prev_cluster = cluster
        docs_chosen = []
        if cluster == None:
            return docs_chosen, None
        for key, value in self.mapping.iteritems():
            if value == cluster:
                docs_chosen.append(key)
        return docs_chosen, cluster


class multi_topic_oracle(doc_cluster_oracle):
    def __init__(self, env, docs_path, threshold=0, window=5,
                d2v_window=10, size=100, epochs=10, clusters=100, quintile=0.2):
        doc_cluster_oracle.__init__(self, env, docs_path, threshold, window, d2v_window, size, epochs, clusters)
        self.quintile = quintile
        self.prev_clusters = []

    def most_common_topics(self, env):
        env.current_idx += self.window
        self.get_trending_cluster(env)
        b = self.trends
        return b

    def active_topics(self, docs):
        self.mapping = self.map_docs(docs)
        return self.mapping.values()
    
    def imp_documents(self, env, docs, round_no):
        self.env.current_idx = env.current_idx
        self.mapping = self.map_docs(docs)
        if round_no <= self.window:
            logger.info("pick a random cluster")
            clusters = [self.mapping[docs[0][0]]]
        else:
            mct = self.most_common_topics(self.env)
            mct = mct.most_common()
            fraction = int(len(mct)*self.quintile) + 1
            mct = mct[:fraction]
            clusters = [cluster for cluster, _ in mct]
        docs_chosen = []
        active_clusters = []
        if clusters == None:
            return docs_chosen, None
        for doc, cluster in self.mapping.iteritems():
            if cluster in clusters or cluster in self.prev_clusters:
                docs_chosen.append(doc)
                if cluster not in active_clusters:
                    active_clusters.append(cluster)
        self.prev_clusters = active_clusters
        return docs_chosen, active_clusters
