import re
import os
import hashlib
import argparse
import logging
from lib.utils import read_file_by_line, get_hex_hash, fetch_files_in_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def match_files_with_urls(urls, src_path, dest_path):
    for i, url in enumerate(urls):
        sha1sum = get_hex_hash(url)
        fname = sha1sum+'.story'
        files = fetch_files_in_dir(dest_path) 
        date = re.findall('^http://web.archive.org/web/(\S{8})', url)[0]
        date = date.replace('/', '-')
        count = len([f for f in files if f.startswith(date)])
        newfilename = date +'-' + str(count) + '.story'
        if i%1000 == 0:
            logger.info("{} files moved".format(i))
        os.system('mv %s %s' %(os.path.join(src_path, fname), \
                               os.path.join(dest_path, newfilename)))
        
def main(file_path, src_path, dest_path):
    urls = [line.strip('\n') for line in read_file_by_line(file_path)]
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    match_files_with_urls(urls, src_path, dest_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--src-path')
    argparser.add_argument('--dest-path')
    argparser.add_argument('--url-file')

    args = argparser.parse_args()
    #print args.url_file, args.src_path, args.dest_path
    main(args.url_file, args.src_path, args.dest_path)
    
    
        
