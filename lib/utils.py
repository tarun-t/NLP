import re
import os
import hashlib
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# acceptable ways to end a sentence
# unicode
dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]

def read_file_by_line(file_path):
    logger.info("Reading {}".format(file_path))	
    with open(file_path, 'r') as f:
        return f.readlines()

def read_text_file(file_path):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  # CNN/DML specific
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def fetch_files_in_dir(dir_path):
    logger.info("Fetching files from {}".format(dir_path))
    return [f for f in os.listdir(dir_path) \
            if os.path.isfile(os.path.join(dir_path, f))]

def get_hex_hash(s):
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

        
