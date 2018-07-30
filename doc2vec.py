import logging
import os

from random import shuffle
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim import utils
from sklearn.externals import joblib

from preprocess_cnn_dmm_data import fetch_articles_from_data, tokenize_articles
from preprocess_cnn_dmm_data import remove_stop_words
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS_DIR = os.path.abspath('models')
DATA_PATH = os.path.abspath('train_data')

class doc2vec(object):
    def __init__(self, docs_path, epochs=10, size=100, window=10):
        '''
        Input:
            docs+path: absolute path to data files
            epochs: no. of epochs in doc2vec
            size: size of d2v vectors
            window: window length in d2v
        Functionality:
            Preprocesess the text articles and trains a d2v model.
            These articles and models are stored in MODELS_DIR
        '''
        self.docs_path = docs_path
        self.epochs = epochs
        self.size = size
        self.window = window
        self.d_prefix = ''.join(self.docs_path.split('/'))
        self.m_prefix = ''.join(self.docs_path.split('/'))+'E'+str(self.epochs)+'S'+str(self.size)+'W'+str(self.window)
        self.model_name = 'doc2vec'+self.m_prefix+'.d2v'
        try:
            logger.info('Loading docs...')
            self.tagged_docs = joblib.load(os.path.join(MODELS_DIR, self.d_prefix))
            logger.info('Finished loading docs')
        except:
            self.docs = fetch_articles_from_data(self.docs_path)
            self.docs = list(tokenize_articles(self.docs))
            self.docs = remove_stop_words(self.docs)
            self.tagged_docs = [TaggedDocument(doc, [idx]) for idx, doc in enumerate(self.docs)]
            joblib.dump(self.tagged_docs, os.path.join(MODELS_DIR, self.d_prefix))
        self.train_model()

    def train_model(self):
        try:
            model = Doc2Vec.load(os.path.join(MODELS_DIR, self.model_name))
        except:
            model = Doc2Vec(dm=0, min_count=20, window=self.window, size=self.size, sample=1e-5, workers=7)
            model.build_vocab(self.tagged_docs)
            model.train(self.tagged_docs, total_examples=model.corpus_count, epochs=self.epochs)
            model.save(os.path.join(MODELS_DIR, self.model_name))
        self.model = model
