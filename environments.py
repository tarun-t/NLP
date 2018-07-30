import os
import numpy as np

from lib.cnn_dmm_utils import docs_on_day

class ConstDocs(object):
    def __init__(self, corpus, no_docs=20):
        '''
            Input:
                corpus: list of articles
                no_docs: no. of docs in each step
        '''
        self.no_docs = no_docs
        self.corpus = corpus
        self.current_idx = 0

    def get(self):
        '''
            Returns current list of docs
        '''
        files = []
        files = self.corpus[self.current_idx:self.current_idx+self.no_docs]
        files = [(self.current_idx+i, f) for i, f in enumerate(files)]
        return files

    def update(self):
        '''
            Move to next slice
        '''
        self.current_idx += self.no_docs

    def prev(self, n):
        '''
            Input: 
                n: no. of days to look back
            Returns a lost of docs
        '''
        files = []
        if (self.current_idx - n*self.no_docs) >= 0:
            files =  self.corpus[self.current_idx-n*self.no_docs:self.current_idx - (n-1)*self.no_docs]
        files = [(self.current_idx - (n*self.no_docs) + i, f) for i, f in enumerate(files)]
        return files


class DocsByDate(object):
    def __init__(self, corpus, dir_path='train_data'):
        '''
            Input:
                corpus: list of articles
                dir_path: path to docs; name of docs should be YYYYMMDD-*
        '''
        self.corpus = corpus
        self.dir_path = dir_path
        self.docs = docs_on_day(self.dir_path)
        self.doc_list = np.cumsum(self.docs)
        self.current_idx = 0

    def get(self):
        '''
            Returns todays docs
        '''
        files = []
        if self.current_idx >= len(self.doc_list):
            return files
        if self.current_idx == 0:
            files = self.corpus[0:self.doc_list[self.current_idx]]
        elif self.current_idx + 1 == len(self.doc_list):
            files = self.corpus[self.doc_list[self.current_idx]:]
        else:
            files = self.corpus[self.doc_list[self.current_idx]:self.doc_list[self.current_idx+1]]
        files = [(self.doc_list[self.current_idx]+i-1, f) for i, f in enumerate(files)]
        return files

    def update(self):
        '''
            Updates the date to next closest data
        '''
        self.current_idx +=1

    def prev(self, n):
        files = []
        if (self.current_idx - n) >= 0:
            files = self.corpus[self.doc_list[self.current_idx-n]:self.doc_list[self.current_idx-n+1]]
        files = [(self.doc_list[self.current_idx-n]+i, f) for i, f in enumerate(files)]
        return files
