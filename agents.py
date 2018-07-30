import logging
import sys, warnings
import numpy as np

from collections import defaultdict

warnings.filterwarnings('error')

class ExpWeights(object):
    def __init__(self, eta=0.1, experts=10):
        self.experts = experts
        self.eta = eta
        self.expert_weights = np.ones(self.experts)
        self.actions = 2

    def choose_topic(self):
        z = np.random.uniform()
        cum_prob = 0.0
        probs = self.expert_weights/np.sum(self.expert_weights)
        for expert in range(self.experts):
            prob = probs[expert]
            cum_prob += prob
            if cum_prob > z:
                return expert
        return self.experts - 1

    def get_lr(self, round_no):
        return self.eta

    def update(self, topic_id, round_no):
        loss = np.ones(self.experts)
        loss[topic_id] = 0
        eta = self.get_lr(round_no)
        self.expert_weights *= np.exp(-eta*loss)


class FSF(object):
    def __init__(self, eta=0.1, experts=2, alpha=0.1):
        self.experts = experts
        self.eta = eta
        self.alpha = alpha
        self.expert_weights = np.ones(self.experts)/experts
        self.actions = 2

    def choose_topic(self):
        z = np.random.uniform()
        cum_prob = 0.0
        probs = self.expert_weights/np.sum(self.expert_weights) 
        for expert in range(self.experts):
            prob = probs[expert]
            cum_prob += prob
            if cum_prob > z:
                return expert
        return self.experts - 1

    def get_lr(self):
        return self.eta

    def update(self, topic_id, round_no):
        loss = np.ones(self.experts)
        loss[topic_id] = 0
        v = self.expert_weights*np.exp(-self.eta*loss)
        self.expert_weights = self.alpha*np.sum(v)/self.experts + (1-self.alpha)*v


class MultiTopicTracking(object):
    def __init__(self, eta=0.1, alpha=0.1, topics=50, tracking=False):
        self.topics = topics
        self.eta = eta
        self.alpha = alpha
        if not tracking:
            self.experts = [ExpWeights(self.eta, experts=2) for i in range(self.topics)]
        else:
            self.experts = [FSF(self.eta, experts=2, alpha=self.alpha) for i in range(self.topics)]

    def choose_topics(self):
        topics = []
        for i, expert in enumerate(self.experts):
            if expert.choose_topic() == 1:
                topics.append(i)
        return topics

    def update(self, topics, round_no):
        for i, expert in enumerate(self.experts):
            if i in topics:
                expert.update(1, round_no)
            else:
                expert.update(0, round_no)
