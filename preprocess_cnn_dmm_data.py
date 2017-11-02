import os
import re
import logging
import gensim

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.utils import lemmatize
from gensim.corpora import Dictionary
from gensim.models import HdpModel, LdaModel
from gensim.models import Phrases

from lib.cnn_dmm_utils import get_art_and_summary, split_data
from lib.utils import fetch_files_in_dir

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIR_PATH = 'cnn_data'
MODELS_DIR = 'models/'
STOPWORD_FILE = 'stopwords.txt'
STOPWORD_FILE = os.path.abspath(STOPWORD_FILE)
DIR_PATH = os.path.abspath(DIR_PATH)
MODELS_DIR = os.path.abspath(MODELS_DIR)
TRAIN_DIR_PATH = os.path.abspath('train_data')
VALIDATION_DIR_PATH = os.path.abspath('validation_data')
TEST_DIR_PATH = os.path.abspath('test_data')

logger.info('Creating data directories')
split_data(DIR_PATH, TRAIN_DIR_PATH, VALIDATION_DIR_PATH, TEST_DIR_PATH)

def fetch_articles_from_data(data_path):
    train_stories = fetch_files_in_dir(data_path)
    articles = []
    for story in sorted(train_stories):
        art, summ = get_art_and_summary(os.path.join(data_path, story))
        if not len(art):
            logger.warning('Article length 0 for {}'.format(story))
            continue
        articles.append(art)
    return articles


def tokenize_articles(articles):
    logger.info('tokenizing articles')     
    texts = [[] for i in articles]
    for idx, art in enumerate(articles):
        texts[idx] = gensim.utils.simple_preprocess(' '.join(art), deacc=True, min_len=3)
    return texts

def add_phrases(articles, min_count=40):
    print articles[0]
    logger.info('adding bigrams')
    bigram = Phrases(articles, min_count=min_count)
    articles = [bigram[line] for line in articles]
    return articles


def read_stop_words():
    with open(STOPWORD_FILE, 'r') as f:
        stopwords = f.readlines()
    stopwords = [word.strip('\n') for word in stopwords]
    return [word.decode('utf-8') for word in stopwords]

def remove_stop_words(articles, stop_words=[]):
    logger.info('removing stopwords')
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    stopwords = set(stopwords) | set(stop_words)
    articles = [[token for token in article if token not in stopwords and not token.isdigit()] for article in articles]
    return articles


def lemmatize_articles(articles):
    logger.info('lemmatizing articles')
    print articles[0]
    articles = [[word.split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=3)] for line in articles]
    return articles

def create_dict(articles):
    dictionary = Dictionary(articles)
    filter_extreme_values(dictionary)
    return dictionary

def save_dict(dictionary, dict_name='cnn_dict.dict'):
    dictionary.save(os.path.join(MODELS_DIR, dict_name))

def construct_dict(articles, dictionary_path=None):
    try:
        dictionary = gensim.corpora.Dictionary.load(dictionary_path)
    except:
        dictionary = create_dict(articles)
    return dictionary

def filter_extreme_values(dictionary):
    dictionary.filter_extremes(no_below=40)


def create_corpus(articles, dictionary):
    corpus = [dictionary.doc2bow(article) for article in articles]
    return corpus

def save_corpus(corpus, corpus_name='cnn_corpus.mm'):
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, corpus_name), corpus)

def construct_corpus(articles, dictionary, corpus_path=None):
    try:
        corpus = gensim.corpora.mmcorpus.MmCorpus(corpus_path)
    except:
        corpus = create_corpus(articles, dictionary)
    return corpus

def preprocess_articles(articles):
    articles = list(tokenize_articles(articles))
    print len(articles)
    articles = remove_stop_words(articles, stop_words=[])
    print len(articles)
    articles = add_phrases(articles)
    articles = lemmatize_articles(articles)
    print len(articles)
    return articles
