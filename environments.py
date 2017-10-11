import os

class ConstDocs(object):
    def __init__(self, corpus, no_docs=20):
        self.no_docs = no_docs
        self.corpus = corpus
        self.current_idx = 0

    def get(self):
        files = []
        files = self.corpus[self.current_idx:self.current_idx+self.no_docs]
        files = [(self.current_idx+i, f) for i, f in enumerate(files)]
        return files

    def update(self):
        self.current_idx += self.no_docs

    def prev(self, n):
        files = []
        # TODO:Update logic
        if (self.current_idx - n*self.no_docs) >= 0:
            files =  self.corpus[self.current_idx-n*self.no_docs:self.current_idx - (n-1)*self.no_docs]
        files = [(self.current_idx - (n*self.no_docs) + i, f) for i, f in enumerate(files)]
        return files
        
