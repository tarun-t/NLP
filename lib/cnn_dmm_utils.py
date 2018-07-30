import re
import os
import subprocess
import logging

from utils import read_text_file, fix_missing_period, fetch_files_in_dir

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s:%(message)s', datefmt='%I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def get_art_and_summary(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    #lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    # article = ' '.join(article_lines)

    # Make summary into a signle string, putting <s> and </s> tags around the sentences
    # summary = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    logger.info("Article and summary for {}".format(story_file))
    return article_lines, highlights


def chunk_list(big_list, n):
    """Yield successive n-sized chunks from big_list."""
    for i in range(0, len(big_list), n):
        yield big_list[i:i + n]

def move_files(src, dest):
    try:
        os.system('mv {} {}'.format(' '.join(src), dest))
    except OSError as e:
        logger.error(e)

def docs_on_day(dir_path):
    docs = sorted(fetch_files_in_dir(dir_path))
    doc_list = []
    prev = None
    count = 0
    for i, doc in enumerate(docs):
        try:
            date = doc.split('.')[0].split('-')[0]
        except:
            raise "File Pattern mismatch"
        # ignore when article length is zero
        art, _ = get_art_and_summary(os.path.join(dir_path, doc))
        if not len(art):
            continue
        if prev != date:
            prev = date
            doc_list.append(count+1)
            count = 0
        elif len(docs)-1 == i:
            doc_list.append(count+2)
        else:
            count += 1 
    return doc_list

def split_data(dir_path, train_dir, valid_dir, test_dir):
    logger.info('Directory containing data {}'.format(dir_path))
    if not len(os.listdir(dir_path)):
        logger.info('No data')
        return
    # check dates when new data is added
    train_data = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if re.search('(200[7-9]|201[0-3]).*', f)]
    validation_data = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if re.search('20140[1-6].*', f)]
    test_data = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if re.search('(20140[7-9]|20141|2015).*', f)]

    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(valid_dir): os.mkdir(valid_dir)
    if not os.path.isdir(test_dir): os.mkdir(test_dir)
    
    logger.info('Creating Training data ...')
    # split into chunks to avoid os errors
    for small_list in chunk_list(train_data, 100):
        move_files(small_list, train_dir)

    logger.info('Creating Validation data ...')
    for small_list in chunk_list(validation_data, 100):
        move_files(small_list, valid_dir)

    logger.info('Creating Test data ...')
    for small_list in chunk_list(test_data, 100):
        move_files(small_list, test_dir)
