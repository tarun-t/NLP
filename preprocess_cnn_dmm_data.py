import os
import logging
import gensim

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
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
    for story in train_stories:
        art, summ = get_art_and_summary(os.path.join(data_path, story))
        if not len(art):
            logger.warning('Article length 0 for {}'.format(story))
            continue
        articles.append(art.decode('utf-8'))

    return articles


def tokenize_articles(articles):
    tokenizer = RegexpTokenizer('\\w+')
    for idx, art in enumerate(articles):
        articles[idx] = tokenizer.tokenize(articles[idx])

    return articles


def add_phrases(articles, min_count=40):
    bigram = Phrases(articles, min_count=min_count)
    update_stopwords =set([])
    for idx in range(len(articles)):
        for token in bigram[articles[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                articles[idx].append(token)
                update_stopwords = set(token.split('_')) | update_stopwords
    print update_stopwords
    articles = [[token for token in article if token not in update_stopwords] for article in articles]
    return articles


def remove_stop_words(articles, stop_words=[]):
    stopwords = set(stopwords.words('english')) | set(stop_words)
    articles = [[token for token in article if len(token) > 2 and token not in stopwords and not token.isdigit()] for article in articles]
    return articles


def lemmatize_articles(articles):
    lemmatizer = WordNetLemmatizer()
    lemmatized_articles = []
    for art in articles:
        lemm_art = []
        for token in art:
            try:
                lemm_art.append(lemmatizer.lemmatize(token))
            except UnicodeError:
                pass

        lemmatized_articles.append(lemm_art)

    return lemmatized_articles

def create_dict(articles, dict_name):
    dictionary = Dictionary(articles)
    filter_extreme_values(dictionary)
    dictionary.save(os.path.join(MODELS_DIR, dict_name))
    return dictionary

def construct_dict(articles, dictionary_path=None, dict_name='cnn_dict.dict', create_new=False):
    if create_new:
        return create_dict(articles, dict_name)
    try:
        dictionary = corpora.Dictionary.load(dictionary_path)
    except:
        dictionary = create_dict(articles, dict_name)
    return dictionary


def filter_extreme_values(dictionary):
    dictionary.filter_extremes(no_below=40)


def create_corpus(articles, dictionary, corpus_name):
    corpus = [dictionary.doc2bow(article) for article in articles]
    gensim.corpora.mmcorpus.MmCorpus.serialize(os.path.join(MODELS_DIR, corpus_name), corpus)
    return corpus

def construct_corpus(articles, dictionary, corpus_path=None, corpus_name='cnn_corpus.mm', create_new=False):
    if create_new:
        return create_corpus(articles, dictionary, corpus_name)
    try:
        corpus = gensim.corpora.mmcorpus.MmCorpus(corpus_path)
    except:
        corpus = create_corpus(articles, dictionary, corpus_name)
    return corpus