import re
import os
import logging

from utils import read_text_file, fix_missing_period

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def get_art_and_summary(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods;
  # consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

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
  article = ' '.join(article_lines)

  # Make summary into a signle string, putting <s> and </s> tags around the sentences
  summary = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  logger.info("Article and summary for {}".format(story_file))
  return article, summary


def split_data(dir_path):
  train_data = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if re.search('(200[7-9]|201[0-4])*', f)]
  validation_data = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if re.search('20140[1-6]*', f)]
  test_data = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f not in train_data and validation_data]

  if not os.isdir('train_data'): os.mkdir('train_data')
  if not os.isdir('validation_data'): os.mkdir('validation_data')
  if not os.isdir('test_data'): os.mkdir('test_data')

  os.system('mv {} train_data/'.format(' '.join(train_data)))
  os.system('mv {} validation_data/'.format(' '.join(validation_data)'))
  os.system('mv {} test_data/'.format(' '.join(test_data))) 
