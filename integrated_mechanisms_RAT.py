from collections import defaultdict
import nltk
from nltk.corpus import semcor
import json
import csv
import os.path
from sentence_long_term_memory import sentenceLTM
from nltk.corpus import wordnet as wn_corpus
from sentence_cooccurrence_activation import *
from n_gram_cooccurrence.google_ngrams import *
import math
import numpy


class RATAgentTemplate:

    def to_string_id(self):
        result = ''
        # result += '_' + str(self.blah)
        # and so on
        return result

    def initialize(self):
        pass

    def new_trial_callback(self):
        pass

    def get_stopwords(self):
        """ Returns list of stopwords"""
        with open('./nltk_english_stopwords', "r") as stopwords_file:
            lines = stopwords_file.readlines()
            stopwords = []
            for l in lines:
                stopwords.append(l[:-1])
        return stopwords

    def guess(self, context1, context2, context3):
        pass

    def setup_run(self):
        # Will be task dependent.
        pass


class AgentSpreading(RATAgentTemplate):
    """ Spreading agent for RAT task. """

    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0, bounded=False):
        super().__init__()
        self.source = source
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.spreading = spreading  # T/F on whether the network activation will spread or not.
        self.bounded = bounded
        self.sem_rel_dict = self.get_sem_rel_dict()
        self.network = self.create_sem_network()
        self.stopwords = self.get_stopwords()

    def to_string_id(self):
        result = 'AgentSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_spreading_' + str(self.spreading)
        result += '_source_' + str(self.source)
        result += '_bounded_' + str(self.bounded)
        return result

    def new_trial_callback(self):
        self.clear_network(start_time=0)

    def get_sem_rel_dict(self):
        """ Gets dictionary of semantic relations from file."""
        sem_rel_link = "./semantic_relations_lists/" + self.source + "_sem_rel_dict.json"
        if os.path.isfile(sem_rel_link):
            sem_rel_file = open(sem_rel_link, "r")
            sem_rel_dict = json.load(sem_rel_file)
        else:
            raise ValueError()
        return sem_rel_dict

    def filter_sem_rel_dict(self, sem_rel_dict):
        """
        Filters the sem rel dict for stopwords to ensure that all words are valid.
        Parameters:
            sem_rel_dict (dictionary): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
        Returns:
            (dict) filtered semantic relations dictionary.
        """
        keys = sem_rel_dict.keys()
        for key in keys:
            rels = sem_rel_dict[key]
            words = [word for word in rels if word.lower() not in self.stopwords]
            sem_rel_dict[key] = words
        return sem_rel_dict

    def create_sem_network(self):
        """ Creates a semantic network from the semantic relations list. """
        if self.bounded:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                                BoundedActivation(
                                    ltm,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = list(self.sem_rel_dict.keys())
        for word in keys:
            assocs = [elem.upper() for elem in self.sem_rel_dict[word]]
            network.store(mem_id=word.upper(),
                          time=1,
                          activate=False,
                          assocs=assocs)
        return network

    def clear_network(self, start_time=1):
        """
            Clears the semantic network by resetting activations to a certain "starting time".
            Parameters:
                sem_network (sentenceLTM): Network to clear.
                start_time (int): The network will be reset so that activations only at the starting time and before the
                    starting time remain.
            Returns:
                sentenceLTM: Cleared semantic network.
            """
        activations = self.network.activation.activations
        if start_time > 0:
            activated_words = activations.keys()
            for word in activated_words:
                activations[word] = [act for act in activations[word] if act[0] <= start_time]
            self.network.activation.activations = activations
        elif start_time == 0:
            self.network.activation.activations = defaultdict(list)
        else:
            raise ValueError(start_time)

    def get_distribution(self, context1, context2, context3):
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth)
        elements = sorted(set(self.network.activation.activations.keys()))
        dist = {}
        for elem in elements:
            if elem in context_list:
                continue
            act = self.network.get_activation(mem_id=elem, time=3)
            if act is not None:
                dist[elem] = act
                print(elem, act)
        return dist

    def guess(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth)
        max_act = -float("inf")
        guesses = []
        elements = sorted(set(self.network.activation.activations.keys()))
        for elem in elements:
            if elem in context_list:
                continue
            elem_act = self.network.get_activation(mem_id=elem, time=3)
            if elem_act is None:
                continue
            elif elem_act > max_act:
                max_act = elem_act
                guesses = [elem]
            elif elem_act == max_act:
                guesses.append(elem)
        return guesses


class AgentCooccurrence(RATAgentTemplate):
    """Cooccurrence agent for Remote Associates Test"""

    def __init__(self, ngrams=GoogleNGram('~/ngram')):
        self.ngrams = ngrams
        self.cooc_cache = self.get_cooccurrence_cache_dict()
        self.stopwords = self.get_stopwords()

    def to_string_id(self):
        result = "AgentCooccurrence"
        return result

    def get_cooccurrence_cache_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/ngrams_cooccurrence_cache.json"))
        cooc_cache_dict = defaultdict(list)
        for entry in cooc_cache:
            key = tuple([entry[0][0].upper(), entry[0][1].upper(), entry[0][2].upper()])
            cooc_elements = entry[1]
            vals = []
            for elem in cooc_elements:
                val = elem[0]
                counts = elem[1]
                vals.append([val, counts])
            cooc_cache_dict[key] = vals
        return cooc_cache_dict

    def get_word_counts(self, word):
        """
        Returns the number of times the word occurred in the ngrams corpus.
        Assumes merge_variants (whether different capitilizations should be considered the same word) to be true.
        Parameters:
            word (string): Word to get the counts of.
        Returns:
            (int): counts
        """
        return self.ngrams.get_ngram_counts(word)[word]

    def get_all_word_cooccurrences(self, word):
        """
        Finds all words that cooccur with a word of interest.
        Parameters:
            word (string): word of interest.
        Returns:
            (list) ordered list (most to least # of times word occurs) of tuples formatted as
        (word, # times word occcurred) for words that cooccur with the input word.
         """
        return self.ngrams.get_max_probability(word)

    def get_conditional_probability(self, word, context):
        """
        Gets the conditional probability of seeing a particular word given the context.
        Parameters:
            target (varies): The word of interest.
            base (varies): The context.
        Returns:
            (float) decimal conditional probability.
        """
        return self.ngrams.get_conditional_probability(base=context, target=word)

    def get_distribution(self, context1, context2, context3):
        """
        """
        dist = {}
        joint_cooc_set = self.cooc_cache[tuple([context1, context2, context3])]
        print(joint_cooc_set)
        for elem in joint_cooc_set:
            cooc_word = elem[0]
            if cooc_word.lower() in self.stopwords:
                continue
            dist[cooc_word] = elem[1]
        return dist

    def guess(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        joint_cooc_set = self.cooc_cache[tuple([context1, context2, context3])]
        if len(joint_cooc_set) == 0:
            return []
        elif len(joint_cooc_set) == 1:
            return list(joint_cooc_set)[0][0]
        else:
            max_cond_prob = -float("inf")
            max_elems = []
            for elem in list(joint_cooc_set):
                cooc_word = elem[0]
                if cooc_word.lower() in self.stopwords:
                    continue
                joint_cond_prob = elem[1]
                if joint_cond_prob > max_cond_prob:
                    max_cond_prob = joint_cond_prob
                    max_elems = [cooc_word]
                elif joint_cond_prob == max_cond_prob:
                    max_elems.append(cooc_word)
            return max_elems


class AgentCoocThreshSpreading(AgentSpreading):
    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 ngrams=GoogleNGram('~/ngram'), threshold=0.0, bounded=False):
        """
        Parameters:
            sem_rel_dict (dictionary): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
            stopwords (list): A list of stopwords - common words to not include semantic relations to.
            ngrams (class): Instance of the GoogleNGram class.
            spreading (bool): Whether to include the effects of spreading in creating the semantic network.
            clear (string): How often to clear the network. Possible values are "never", "trial",
                indicating that the network is never cleared, or cleared after each RAT trial.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(source, spreading, activation_base, decay_parameter, constant_offset, bounded)
        self.stopwords = self.get_stopwords()
        self.threshold = threshold
        self.ngrams = ngrams
        self.sem_rel_dict = self.get_thresh_sem_rel_dict()
        self.network = self.create_sem_network()

    def to_string_id(self):
        result = "AgentCoocThreshSpreading"
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_threshold_' + str(self.threshold)
        result += '_spreading_' + str(self.spreading)
        result += '_source_' + str(self.source)
        result += '_bounded_' + str(self.bounded)
        return result

    def get_stopwords(self):
        with open('./nltk_english_stopwords', "r") as stopwords_file:
            lines = stopwords_file.readlines()
            stopwords = []
            for l in lines:
                stopwords.append(l[:-1])
        return stopwords

    def get_thresh_sem_rel_dict(self):
        """ Getting the thresholded semantic relations dictionary from file and/or creating it"""
        sem_rel_path = "./semantic_relations_lists/" + self.source + "_thresh_"
        if self.threshold != 0:
            sem_rel_path += str(self.threshold) + "_"
        sem_rel_path += "sem_rel_dict.json"
        if not os.path.isfile(sem_rel_path):
            sem_rel_dict = self.get_sem_rel_dict()
            adjusted_sem_rel_dict = self.adjust_sem_rel_dict(sem_rel_dict)
            file = open(sem_rel_path, 'w')
            json.dump(adjusted_sem_rel_dict, file)
            file.close()
            return sem_rel_dict
        sem_rel_dict = json.load(open(sem_rel_path))
        return sem_rel_dict

    def get_cooccurring_words(self, word):
        """
        Gets the words that cooccur with a given input word in google ngrams
        Parameters:
            word (string): word of interest
        Returns:
             (list) ordered list (most to least # of times word occurs) of tuples formatted as
             (word, # times word occcurred) for words that cooccur with the input word"""
        cooc_words_counts = self.ngrams.get_max_probability(word)
        cooc_words = [word[0] for word in cooc_words_counts if word not in self.stopwords]
        return cooc_words

    def adjust_sem_rel_dict(self, sem_rel_dict):
        """
        Adjusts the semantic relations dictionary to only include words that also cooccur with each word (as key).
        Parameters:
            sem_rel_dict (dict): A dictionary with all semantic relations of interest - from SWOWEN, SFFAN, or both.
            save_dict (str): A link to save the adjusted dictionary to. If none, nothing is saved.
        Returns:
            (dict) Cooccurrence adjusted semantic relations dictionary.
        """
        thresh_sem_rel_dict = defaultdict(list)
        counter = 0
        leng = len(list(sem_rel_dict.keys()))
        for word_key in sorted(list(sem_rel_dict.keys())):
            counter += 1
            #if counter % 50 == 0:
                #(counter, "out of", leng, "converted")
            if word_key.lower() in self.stopwords or word_key.count(" ") > 0:
                continue  # Making sure we're only looking at bigrams
            rels = sem_rel_dict[word_key]  # has all different relations to target word
            new_rels = []
            for rel in rels:  # going through words corresponding to each relation
                if rel.lower() in self.stopwords or rel.count(" ") != 0:
                    continue  # Making sure we're only looking at bigrams
                counts = self.ngrams.get_ngram_counts(word_key + " " + rel)[word_key + " " + rel] + \
                         self.ngrams.get_ngram_counts(rel + " " + word_key)[rel + " " + word_key]
                if counts > self.threshold:
                    new_rels.append(rel.upper())
            if new_rels:
                thresh_sem_rel_dict[word_key.upper()] = new_rels
        return thresh_sem_rel_dict

    def create_sem_network(self):
        """ Builds a semantic network. """
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        if self.bounded:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                                BoundedActivation(
                                    ltm,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = sorted(list(self.sem_rel_dict.keys()))
        for word in keys:
            assocs = self.sem_rel_dict[word]
            network.store(mem_id=word.upper(),
                          time=1,
                          spread_depth=spread_depth,
                          activate=False,
                          assocs=assocs)
        return network


class AgentSpreadingThreshCooc(AgentCooccurrence, AgentSpreading):

    def __init__(self, source, activation_base, decay_parameter, constant_offset, ngrams=GoogleNGram('~/ngram')):
        """
        Parameters:
            stopwords (list): A list of stopwords - common words to not include semantic relations to.
            sem_rel_dict (dict): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
            ngrams (class): The google ngrams class.
        """
        super().__init__(ngrams)
        self.source = source
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.ngrams = ngrams
        self.stopwords = self.get_stopwords()
        self.sem_rel_dict = self.filter_sem_rel_dict(self.get_sem_rel_dict())

    def to_string_id(self):
        result = "AgentCoocThreshSpreading"
        result += "_source_" + self.source
        return result

    def new_trial_callback(self):
        pass

    def get_distribution(self, context1, context2, context3):
        """
        """
        dist = {}
        cooc_prob_list = self.cooc_cache[tuple([context1, context2, context3])]
        cooc_prob_dict = {}
        for elem in cooc_prob_list:
            cooc_prob_dict[elem[0]] = elem[1]
        cooc_set = set(cooc_prob_dict.keys())
        # Now threshold based on semantic relations as well
        sem_rel_set1 = set()
        sem_rel_set2 = set()
        sem_rel_set3 = set()
        if context1.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set1 = set([elem.upper() for elem in self.sem_rel_dict[context1.lower()]])
        if context2.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set2 = set([elem.upper() for elem in self.sem_rel_dict[context2.lower()]])
        if context3.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set3 = set([elem.upper() for elem in self.sem_rel_dict[context3.lower()]])
        joint_cooc_set = cooc_set & sem_rel_set1 & sem_rel_set2 & sem_rel_set3
        for cooc_word in list(joint_cooc_set):
            if cooc_word.lower() in self.stopwords:
                continue
            dist[cooc_word] = cooc_prob_dict[cooc_word]
        return dist

    def guess(self, context1, context2, context3):
        """
        Completes one round of the RAT task.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        cooc_prob_list = self.cooc_cache[tuple([context1, context2, context3])]
        cooc_prob_dict = {}
        for elem in cooc_prob_list:
            cooc_prob_dict[elem[0]] = elem[1]
        cooc_set = set(cooc_prob_dict.keys())
        # Now threshold based on semantic relations as well
        sem_rel_set1 = set()
        sem_rel_set2 = set()
        sem_rel_set3 = set()
        if context1.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set1 = set([elem.upper() for elem in self.sem_rel_dict[context1.lower()]])
        if context2.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set2 = set([elem.upper() for elem in self.sem_rel_dict[context2.lower()]])
        if context3.lower() in list(self.sem_rel_dict.keys()):
            sem_rel_set3 = set([elem.upper() for elem in self.sem_rel_dict[context3.lower()]])
        joint_cooc_set = cooc_set & sem_rel_set1 & sem_rel_set2 & sem_rel_set3
        if len(joint_cooc_set) == 0:
            return []
        elif len(joint_cooc_set) == 1:
            return list(joint_cooc_set)
        else:
            max_cond_prob = -float("inf")
            max_elems = []
            for cooc_word in list(joint_cooc_set):
                if cooc_word.lower() in self.stopwords:
                    continue
                joint_cond_prob = cooc_prob_dict[cooc_word]
                if joint_cond_prob > max_cond_prob:
                    max_cond_prob = joint_cond_prob
                    max_elems = [cooc_word]
                elif joint_cond_prob == max_cond_prob:
                    max_elems.append(cooc_word)
            return max_elems


class AgentJointProbability(RATAgentTemplate):
    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 ngrams=GoogleNGram('~/ngram'), bounded=False):
        self.spreading = spreading
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.source = source
        self.ngrams = ngrams
        self.bounded = bounded
        self.cooc_agent = AgentCooccurrence(ngrams)
        self.spreading_agent = AgentSpreading(source, spreading, activation_base, decay_parameter, constant_offset, bounded)

        self.joint_cond_prob_dict = self.get_joint_conditional_probability_dict()

    def to_string_id(self):
        result = "AgentJointProbability"
        result += "_source_" + self.source
        result += '_bounded_' + str(self.bounded)
        return result

    def get_activation_probability(self, word_activation, other_activations, tau, s=0.25):
        """ Gets activation probability for a given element with a specified activation """
        num = math.exp(word_activation / s)
        denom = math.exp(tau / s) + sum(math.exp(act / s) for act in other_activations)
        return num / denom

    def create_joint_distribution(self, cooccurrence_dist, spreading_dist):
        """ Creates a joint distribution, assuming the cooccurrence and spreading distributions are already known"""
        words = set()
        words.update(list(cooccurrence_dist.keys()))
        words.update(list(spreading_dist.keys()))
        joint_dist = defaultdict(float)
        for word in words:
            if word not in cooccurrence_dist.keys() or word not in spreading_dist.keys():
                joint_dist[word] = 0
                continue
            joint_dist[word] = cooccurrence_dist[word] * spreading_dist[word]
        return joint_dist

    def get_joint_conditional_probability_dict(self):
        """ Easier to access conditional probabilities for each word that
            is jointly cooccurrent with the 3 RAT context words"""
        joint_conditional_prob_dict = defaultdict(dict)
        for elem in self.cooc_agent.cooc_cache.keys():
            joint_cooc_rels = self.cooc_agent.cooc_cache[elem]
            sub_dict = defaultdict(int)
            for entry in joint_cooc_rels:
                sub_dict[entry[0]] = entry[1]
            joint_conditional_prob_dict[elem] = sub_dict
        return joint_conditional_prob_dict

    def get_conditional_probability(self, word, context):
        """ Gets conditional probability for ngrams directly from the cooccurrence class.
        Context is the 3 RAT context words given on each trial.
        """
        return self.joint_cond_prob_dict[tuple(context)][word]

    def get_cooccurrence_distribution(self, context1, context2, context3):
        """ Getting a conditional distribution dictionary with keys the words jointly related to the 3 context
        words, and values the normalized conditional probabilities. """
        cond_probs = self.joint_cond_prob_dict[tuple([context1.upper(), context2.upper(), context3.upper()])]
        # Normalizing the probabilities
        total = sum(list(cond_probs.values()))
        for key in cond_probs.keys():
            cond_probs[key] = cond_probs[key] / total
        return cond_probs

    def get_spreading_distribution(self, context1, context2, context3):
        """ Gets normalized distribution of activations resulting from activating the 3 context words in the spreading
         mechanism on the RAT"""
        self.spreading_agent.network.store(context1.upper(), 1)
        self.spreading_agent.network.store(context2.upper(), 1)
        self.spreading_agent.network.store(context3.upper(), 1)
        acts = defaultdict(float)
        for elem in self.spreading_agent.network.activation.activations.keys():
            act = self.spreading_agent.network.get_activation(elem, 2)
            if act is not None and act != 0:
                acts[elem] = act
        act_prob_dist = defaultdict(float)
        act_sum = 0
        other_acts = list(acts.values())
        for elem in acts.keys():
            act_prob_dist[elem] = self.get_activation_probability(acts[elem], other_acts, math.log(3 / 8))
            act_sum += act_prob_dist[elem]
        for elem in act_prob_dist.keys():
            act_prob_dist[elem] = act_prob_dist[elem] / act_sum
        self.spreading_agent.clear_network(0)  # reset the network.
        return act_prob_dist

    def get_distribution(self, context1, context2, context3):
        """
        """
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        dist = self.create_joint_distribution(spreading_dist=spread_dist, cooccurrence_dist=cooc_dist)
        return dist

    def guess(self, context1, context2, context3):
        """ Does one trial of the RAT task"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        joint_probs = self.create_joint_distribution(spreading_dist=spread_dist, cooccurrence_dist=cooc_dist)
        joint_candidates = list(joint_probs.keys())
        #print(joint_probs.items())
        if len(joint_candidates) == 0:
            return []
        elif len(joint_candidates) == 1:
            return joint_candidates[0]
        else:
            max_act = -float("inf")
            guesses = []
            for candidate in joint_candidates:
                act = joint_probs[candidate]
                if act > max_act:
                    max_act = act
                    guesses = [candidate]
                elif act == max_act:
                    guesses.append(candidate)
            return guesses


class AgentAdditiveProbability(AgentJointProbability):
    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 ngrams=GoogleNGram('~/ngram'), bounded=False):

        super().__init__(source, spreading, activation_base, decay_parameter, constant_offset, ngrams, bounded)

    def to_string_id(self):
        result = "AgentAdditiveProbability"
        result += "_source_" + self.source
        result += '_bounded_' + str(self.bounded)
        return result

    def create_joint_distribution(self, cooccurrence_dist, spreading_dist):
        """ Creates a joint distribution, assuming the cooccurrence and spreading distributions are already known.
        Overrides joint distribution function from AgentJointProbability (gets the multiplicative joint probability
         distribution) to get the "additive" joint distribution"""
        words = set()
        words.update(list(cooccurrence_dist.keys()))
        words.update(list(spreading_dist.keys()))
        joint_dist = defaultdict(float)
        for word in words:
            if word not in cooccurrence_dist.keys() and word not in spreading_dist.keys():
                joint_dist[word] = 0
            elif word not in cooccurrence_dist.keys():
                joint_dist[word] = spreading_dist[word]
            elif word not in spreading_dist.keys():
                joint_dist[word] = cooccurrence_dist[word]
            else:
                joint_dist[word] = cooccurrence_dist[word] + spreading_dist[word]
        return joint_dist


class AgentJointVariance(RATAgentTemplate):
    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 ngrams=GoogleNGram('~/ngram'), var_type="stdev", bounded=False):
        self.var_type = var_type
        self.ngrams = ngrams
        self.bounded = bounded
        self.cooc_agent = AgentCooccurrence(ngrams)
        self.source = source
        self.spreading_agent = AgentSpreading(source=source, spreading=spreading, activation_base=activation_base,
                                              decay_parameter=decay_parameter, constant_offset=constant_offset,
                                              bounded=bounded)
        self.stopwords = self.spreading_agent.get_stopwords()
        self.joint_cond_prob_dict = self.get_joint_conditional_probability_dict()

    def to_string_id(self):
        result = "AgentJointVariance"
        result += "_source_" + self.source
        result += "_type_" + self.var_type
        result += '_bounded_' + str(self.bounded)
        return result

    def get_activation_probability(self, word_activation, other_activations, tau, s=0.25):
        """ Gets activation probability for a given element with a specified activation """
        num = math.exp(word_activation / s)
        denom = math.exp(tau / s) + sum(math.exp(act / s) for act in other_activations)
        return num / denom

    def create_joint_distribution(self, cooccurrence_dist, spreading_dist):
        """ Creates a joint distribution, assuming the cooccurrence and spreading distributions are already known"""
        words = set()
        words.update(list(cooccurrence_dist.keys()))
        words.update(list(spreading_dist.keys()))
        joint_dist = defaultdict(float)
        for word in words:
            if word not in cooccurrence_dist.keys() or word not in spreading_dist.keys():
                joint_dist[word] = 0
                continue
            joint_dist[word] = cooccurrence_dist[word] * spreading_dist[word]
        return joint_dist

    def get_distribution_variance(self, distribution):
        """ Gets the spread of the distribution via standard deviation.
            Possible values of var_type:
                "stdev" --> Returns the standard deviation of the distribution.
                "maxdiff" --> Returns the difference between the highest probability and 2nd highest probability items
        """
        vals = sorted(list(distribution.values()))
        if self.var_type == "stdev":
            return numpy.std(vals)
        elif self.var_type == "maxdiff":
            if len(vals) < 2:
                return 0
            return vals[-1] - vals[-2]
        else:
            raise ValueError(self.var_type)

    def get_guesses(self, dist):
        """ Makes guesses based on pre-calculated distributions. """
        max_prob = -float("inf")
        max_guesses = []
        for key in dist.keys():
            prob = dist[key]
            if prob == max_prob:
                max_guesses.append(key)
            elif prob > max_prob:
                max_prob = prob
                max_guesses = [key]
        return max_guesses

    def get_joint_conditional_probability_dict(self):
        """ Easier to access conditional probabilities for each word that
            is jointly cooccurrent with the 3 RAT context words"""
        joint_conditional_prob_dict = defaultdict(dict)
        for elem in self.cooc_agent.cooc_cache.keys():
            joint_cooc_rels = self.cooc_agent.cooc_cache[elem]
            sub_dict = defaultdict(int)
            for entry in joint_cooc_rels:
                sub_dict[entry[0]] = entry[1]
            joint_conditional_prob_dict[elem] = sub_dict
        return joint_conditional_prob_dict

    def get_cooccurrence_distribution(self, context1, context2, context3):
        """ Getting a conditional distribution dictionary with keys the words jointly related to the 3 context
        words, and values the normalized conditional probabilities. """
        cond_probs = self.joint_cond_prob_dict[tuple([context1.upper(), context2.upper(), context3.upper()])]
        # Normalizing the probabilities
        total = sum(list(cond_probs.values()))
        for key in cond_probs.keys():
            cond_probs[key] = cond_probs[key] / total
        return cond_probs

    def get_spreading_distribution(self, context1, context2, context3):
        """ Gets normalized distribution of activations resulting from activating the 3 context words in the spreading
         mechanism on the RAT"""
        self.spreading_agent.network.store(context1.upper(), 1)
        self.spreading_agent.network.store(context2.upper(), 1)
        self.spreading_agent.network.store(context3.upper(), 1)
        acts = defaultdict(float)
        for elem in self.spreading_agent.network.activation.activations.keys():
            act = self.spreading_agent.network.get_activation(elem, 2)
            if act is not None and act != 0:
                acts[elem] = act
        act_prob_dist = defaultdict(float)
        act_sum = 0
        other_acts = list(acts.values())
        for elem in acts.keys():
            act_prob_dist[elem] = self.get_activation_probability(acts[elem], other_acts, math.log(3 / 8))
            act_sum += act_prob_dist[elem]
        for elem in act_prob_dist.keys():
            act_prob_dist[elem] = act_prob_dist[elem] / act_sum
        self.spreading_agent.clear_network(0)  # reset the network.
        return act_prob_dist

    def get_distribution(self, context1, context2, context3):
        """ Does the RAT test in an oracle manner based on whichever method has the most variance"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        cooc_dist_variance = self.get_distribution_variance(cooc_dist)
        spread_dist_variance = self.get_distribution_variance(spread_dist)
        if cooc_dist_variance < spread_dist_variance:
            return spread_dist
        else:
            return cooc_dist

    def guess(self, context1, context2, context3):
        """ Does the RAT test in an oracle manner based on whichever method has the most variance"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        cooc_dist_variance = self.get_distribution_variance(cooc_dist)
        spread_dist_variance = self.get_distribution_variance(spread_dist)
        if cooc_dist_variance < spread_dist_variance:
            guesses = self.get_guesses(spread_dist)
        else:
            guesses = self.get_guesses(cooc_dist)
        return guesses


class AgentMaxProbability(AgentJointProbability):
    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 ngrams=GoogleNGram('~/ngram'), bounded=False):
        super().__init__(source, spreading, activation_base, decay_parameter, constant_offset, ngrams, bounded)

    def to_string_id(self):
        result = "AgentMaxProbability"
        result += "_source_" + self.source
        result += '_bounded_' + str(self.bounded)
        return result

    def get_pairwise_cooc_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/single_ngrams.json"))
        cooc_cache_dict = defaultdict(int)
        for sublist in cooc_cache:
            for elem in sublist:
                cooc_cache_dict[tuple([elem[0][0].upper(), elem[0][1].upper()])] = elem[1]
        return cooc_cache_dict

    def get_distribution(self, context1, context2, context3):
        """ Does one trial of the RAT task"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        if not spread_dist and not cooc_dist:
            return []
        dist = {}
        keys = set(spread_dist.keys())
        keys.update(cooc_dist.keys())
        for key in keys:
            if key in spread_dist.keys() and not key in cooc_dist.keys():
                dist[key] = spread_dist[key]
            elif not key in spread_dist.keys() and key in cooc_dist.keys():
                dist[key] = cooc_dist[key]
            else:
                if cooc_dist[key] > spread_dist[key]:
                    dist[key] = cooc_dist[key]
                else:
                    dist[key] = spread_dist[key]
        return dist

    def guess(self, context1, context2, context3):
        """ Does one trial of the RAT task"""
        context_words = [context1.upper(), context2.upper(), context3.upper()]
        cooc_dist = self.get_cooccurrence_distribution(context_words[0], context_words[1], context_words[2])
        spread_dist = self.get_spreading_distribution(context_words[0], context_words[1], context_words[2])
        if not spread_dist and not cooc_dist:
            return []
        elif not spread_dist:
            max_cooc = max(cooc_dist.values())
            guesses = [k for k, v in cooc_dist.items() if v == max_cooc]
        elif not cooc_dist:
            max_spread = max(spread_dist.values())
            guesses = [k for k, v in spread_dist.items() if v == max_spread]
        else:
            max_spread = max(spread_dist.values())
            spread_guesses = [k for k, v in spread_dist.items() if v == max_spread]
            max_cooc = max(cooc_dist.values())
            cooc_guesses = [k for k, v in cooc_dist.items() if v == max_cooc]
            if max_spread == max_cooc:
                guesses = set()
                guesses.update(spread_guesses)
                guesses.update(cooc_guesses)
                guesses = list(guesses)
            elif max_spread > max_cooc:
                guesses = spread_guesses
            else:
                guesses = cooc_guesses
        return guesses


class AgentCoocWeightedSpreading(AgentSpreading):
    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 bounded=False):
        self.bounded = bounded
        self.cooc_dict = self.get_pairwise_cooc_dict()
        super().__init__(source, spreading, activation_base, decay_parameter, constant_offset, bounded)

    def to_string_id(self):
        result = 'AgentCoocWeightedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_spreading_' + str(self.spreading)
        result += '_source_' + str(self.source)
        result += '_bounded_' + str(self.bounded)
        return result

    def get_pairwise_cooc_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/extended_single_ngrams.json"))
        cooc_cache_dict = defaultdict(int)
        for sublist in cooc_cache:
            for elem in sublist:
                cooc_cache_dict[tuple([elem[0][0].upper(), elem[0][1].upper()])] = elem[1]
        return cooc_cache_dict

    def create_sem_network(self):
        """ Creates a semantic network from the semantic relations list. """
        if self.bounded:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                                CooccurrenceWeightedBoundedActivation(
                                    ltm,
                                    cooc_dict=self.cooc_dict,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                            CooccurrenceWeightedActivation(
                                ltm,
                                cooc_dict=self.cooc_dict,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = list(self.sem_rel_dict.keys())
        for word in keys:
            assocs = [elem.upper() for elem in self.sem_rel_dict[word]]
            network.store(mem_id=word.upper(),
                          time=1,
                          activate=False,
                          assocs=assocs)
        return network


class AgentBoundedSpreading(AgentSpreading):

    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0):
        super().__init__(source, spreading, activation_base, decay_parameter, constant_offset)

    def to_string_id(self):
        result = 'AgentBoundedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_spreading_' + str(self.spreading)
        result += '_source_' + str(self.source)
        return result

    def create_sem_network(self):
        """ Creates a semantic network from the semantic relations list. """
        network = sentenceLTM(
            activation_cls=(lambda ltm:
                            BoundedActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = list(self.sem_rel_dict.keys())
        for word in keys:
            assocs = [elem.upper() for elem in self.sem_rel_dict[word]]
            network.store(mem_id=word.upper(),
                          time=1,
                          activate=False,
                          assocs=assocs)
        return network


class AgentSpreadingSupplementedCooc(AgentCooccurrence):

    def __init__(self, source, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 ngrams=GoogleNGram('~/ngram'), bounded=False):
        super().__init__(ngrams)
        self.spread_agent = AgentSpreading(source, spreading, activation_base, decay_parameter, constant_offset, bounded)
        self.sem_rel_dict = self.spread_agent.get_sem_rel_dict()
        self.cooc_dict = self.get_pairwise_cooc_dict()
        self.source = source

    def to_string_id(self):
        result = "AgentSpreadingSupplementedCooc"
        result += "_source_" + self.source
        return result

    def get_cooc_words(self, word):
        filename = "./n_gram_cooccurrence/n_grams_single_cooccurrence/" + word + ".json"
        if os.path.isfile(filename):
            cooc_words_list = json.load(open(filename))
            cooc_pairs = [pair[0] for pair in cooc_words_list]
            cooc_words = []
            for pair in cooc_pairs:
                if pair[0] == word.upper():
                    cooc_words.append(pair[1])
                elif pair[1] == word.upper():
                    cooc_words.append(pair[0])
            return set(cooc_words)
        else:
            return set()

    def get_pairwise_cooc_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/extended_single_ngrams.json"))
        cooc_cache_dict = defaultdict(int)
        for sublist in cooc_cache:
            for elem in sublist:
                cooc_cache_dict[tuple([elem[0][0].upper(), elem[0][1].upper()])] = elem[1]
        return cooc_cache_dict

    def get_associated_words(self, word):
        rels = []
        if word.lower() in self.sem_rel_dict:
            rels = self.sem_rel_dict[word.lower()]
        return rels

    def get_distribution(self, context1, context2, context3):
        """
        """
        dist = {}
        joint_cooc_set = self.cooc_cache[tuple([context1, context2, context3])]
        # Now getting the words semantically related to each context words...
        # These are now part of our context for the co-occurrence mechanism
        assoc_words = set()
        assoc_words.update([word.upper() for word in self.get_associated_words(context1)])
        assoc_words.update([word.upper() for word in self.get_associated_words(context2)])
        assoc_words.update([word.upper() for word in self.get_associated_words(context3)])
        # so we need to find where the cooccurrence sets of each intersects
        # Getting the words that co-occur with all 3 of the context words...
        joint_cooc_words = set()
        joint_cooc_words.update([pair[0] for pair in joint_cooc_set])
        # Now taking the intersection between each of the spreading-related cooccurrence sets
        for word in assoc_words:
            sem_cooc_words = self.get_cooc_words(word)
            joint_cooc_words = joint_cooc_words & sem_cooc_words
            if joint_cooc_words == set():
                break
        for elem in list(joint_cooc_words):
            cooc_word = elem[0]
            if cooc_word.lower() in self.stopwords:
                continue
            # Calculate joint conditional probability for that element
            joint_cond_prob = 1
            assoc_words.update([context1, context2, context3])
            for context in assoc_words:
                if tuple([context.upper(), elem.upper()]) in self.cooc_dict:
                    joint_cond_prob = joint_cond_prob * self.cooc_dict[tuple([context.upper(), elem.upper()])]
                elif tuple([elem.upper(), context.upper()]) in self.cooc_dict:
                    joint_cond_prob = joint_cond_prob * self.cooc_dict[tuple([elem.upper(), context.upper()])]
                else:
                    joint_cond_prob = 0
                    break
            dist[elem] = joint_cond_prob
        return dist

    def guess(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        joint_cooc_set = self.cooc_cache[tuple([context1, context2, context3])]
        # Now getting the words semantically related to each context words...
        # These are now part of our context for the co-occurrence mechanism
        assoc_words = set()
        assoc_words.update([word.upper() for word in self.get_associated_words(context1)])
        assoc_words.update([word.upper() for word in self.get_associated_words(context2)])
        assoc_words.update([word.upper() for word in self.get_associated_words(context3)])
        # so we need to find where the cooccurrence sets of each intersects
        # Getting the words that co-occur with all 3 of the context words...
        joint_cooc_words = set()
        joint_cooc_words.update([pair[0] for pair in joint_cooc_set])
        # Now taking the intersection between each of the spreading-related cooccurrence sets
        for word in assoc_words:
            sem_cooc_words = self.get_cooc_words(word)
            joint_cooc_words = joint_cooc_words & sem_cooc_words
            if joint_cooc_words == set():
                break
        if len(joint_cooc_words) == 0:
            return []
        elif len(joint_cooc_words) == 1:
            # Only one nonzero candidate to choose from
            return list(joint_cooc_words)[0]
        else:
            # We have multiple nonzero candidates to choose from
            max_cond_prob = -float("inf")
            max_elems = []
            for elem in list(joint_cooc_words):
                cooc_word = elem[0]
                if cooc_word.lower() in self.stopwords:
                    continue
                # Calculate joint conditional probability for that element
                joint_cond_prob = 1
                assoc_words.update([context1, context2, context3])
                for context in assoc_words:
                    if tuple([context.upper(), elem.upper()]) in self.cooc_dict:
                        joint_cond_prob = joint_cond_prob * self.cooc_dict[tuple([context.upper(), elem.upper()])]
                    elif tuple([elem.upper(), context.upper()]) in self.cooc_dict:
                        joint_cond_prob = joint_cond_prob * self.cooc_dict[tuple([elem.upper(), context.upper()])]
                    else:
                        joint_cond_prob = 0
                        break
                # Figure out if the current contender has a high enough conditional probability
                if joint_cond_prob > max_cond_prob:
                    max_cond_prob = joint_cond_prob
                    max_elems = [cooc_word]
                elif joint_cond_prob == max_cond_prob:
                    max_elems.append(cooc_word)
            return max_elems


class AgentCoocSupplementedSpreading(AgentSpreading):
    def __init__(self, source, discount=0.1, spreading=True, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 bounded=False):
        self.cooc_dict = self.get_cooccurrence_cache_dict()
        self.discount = discount
        super().__init__(source, spreading, activation_base, decay_parameter, constant_offset, bounded)

    def to_string_id(self):
        result = "AgentCoocSupplementedSpreading"
        result += "_source_" + self.source
        result += '_bounded_' + str(self.bounded)
        return result

    def get_cooccurrence_cache_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/ngrams_cooccurrence_cache.json"))
        cooc_cache_dict = defaultdict(list)
        for entry in cooc_cache:
            key = tuple([entry[0][0].upper(), entry[0][1].upper(), entry[0][2].upper()])
            cooc_elements = entry[1]
            vals = []
            for elem in cooc_elements:
                val = elem[0]
                vals.append(val)
            cooc_cache_dict[key] = vals
        return cooc_cache_dict

    def create_sem_network(self):
        """ Creates a semantic network from the semantic relations list. """
        if self.bounded:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                                CoocBoundedSupplementedActivation(
                                    ltm,
                                    discount=self.discount,
                                    cooc_dict=self.cooc_dict,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                            CoocSupplementedActivation(
                                ltm,
                                discount=self.discount,
                                cooc_dict=self.cooc_dict,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = list(self.sem_rel_dict.keys())
        for word in keys:
            assocs = [elem.upper() for elem in self.sem_rel_dict[word]]
            network.store(mem_id=word.upper(),
                          time=1,
                          activate=False,
                          assocs=assocs)
        return network

    def get_additional_context(self, context):
        if context in self.cooc_dict:
            return self.cooc_dict[context]
        else:
            return []

    def get_distribution(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        dist = {}
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth)
        additional_context = self.get_additional_context(tuple(context_list))
        for word in additional_context:
            self.network.activation.activate_cooc_word(word, time=2, spread_depth=spread_depth)
        elements = sorted(set(self.network.activation.activations.keys()))
        for elem in elements:
            if elem in context_list:
                continue
            dist[elem] = self.network.get_activation(mem_id=elem, time=3)
        return dist

    def guess(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth)
        additional_context = self.get_additional_context(tuple(context_list))
        for word in additional_context:
            self.network.activation.activate_cooc_word(word, time=2, spread_depth=spread_depth)
        max_act = -float("inf")
        guesses = []
        elements = sorted(set(self.network.activation.activations.keys()))
        for elem in elements:
            if elem in context_list:
                continue
            elem_act = self.network.get_activation(mem_id=elem, time=3)
            if elem_act is None:
                continue
            elif elem_act > max_act:
                max_act = elem_act
                guesses = [elem]
            elif elem_act == max_act:
                guesses.append(elem)
        return guesses


class AgentCoocExpandedSpreading(RATAgentTemplate):

    def __init__(self, ngrams=GoogleNGram('~/ngram'), source='SFFAN', spreading=True, activation_base=2,
                 decay_parameter=0.05, constant_offset=0, bounded=False, cooc_depth=1, threshold=0):
        self.network = None
        self.ngrams = ngrams
        self.source = source
        self.spreading = spreading
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.bounded = bounded
        self.cooc_depth = cooc_depth
        self.threshold = threshold
        self.create_blank_network()
        self.semantic_relations_dict = self.get_sem_rel_dict()
        self.cooc_dict = self.get_pairwise_cooc_dict()

    def to_string_id(self):
        result = 'AgentCoocExpandedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_spreading_' + str(self.spreading)
        result += '_source_' + str(self.source)
        result += '_bounded_' + str(self.bounded)
        result += '_coocdepth_' + str(self.cooc_depth)
        result += '_threshold_' + str(self.threshold)
        return result

    def get_pairwise_cooc_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/extended_single_ngrams.json"))
        cooc_cache_dict = defaultdict(list)
        for sublist in cooc_cache:
            for elem in sublist:
                if elem[1] > self.threshold:
                    cooc_cache_dict[elem[0][0].upper()].append(elem[0][1].upper())
                    cooc_cache_dict[elem[0][1].upper()].append(elem[0][0].upper())
        return cooc_cache_dict

    def get_sem_rel_dict(self):
        """ Gets dictionary of semantic relations from file."""
        sem_rel_link = "./semantic_relations_lists/" + self.source + "_sem_rel_dict.json"
        if os.path.isfile(sem_rel_link):
            sem_rel_file = open(sem_rel_link, "r")
            sem_rel_dict = json.load(sem_rel_file)
        else:
            raise ValueError()
        return sem_rel_dict

    def filter_sem_rel_dict(self, sem_rel_dict):
        """
        Filters the sem rel dict for stopwords to ensure that all words are valid.
        Parameters:
            sem_rel_dict (dictionary): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
        Returns:
            (dict) filtered semantic relations dictionary.
        """
        keys = sem_rel_dict.keys()
        for key in keys:
            rels = sem_rel_dict[key]
            words = [word for word in rels if word.lower() not in self.stopwords]
            sem_rel_dict[key] = words
        return sem_rel_dict

    def create_sem_network(self):
        """ Creates a semantic network from the semantic relations list. """
        if self.bounded:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                                BoundedActivation(
                                    ltm,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            network = sentenceLTM(
                activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        keys = list(self.sem_rel_dict.keys())
        for word in keys:
            assocs = [elem.upper() for elem in self.sem_rel_dict[word]]
            network.store(mem_id=word.upper(),
                          time=1,
                          activate=False,
                          assocs=assocs)
        return network

    def create_trial_network(self, words):
        if self.bounded:
            self.network = sentenceLTM(
                activation_cls=(lambda ltm:
                                BoundedActivation(
                                    ltm,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            self.network = sentenceLTM(
                activation_cls=(lambda ltm:
                            SentenceCooccurrenceActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        curr_words = words
        prev_words = set()
        next_words = set()
        distance = 0  # Counter for figuring out how far away words are from the context/target words
        while curr_words:
            distance += 1
            for word in curr_words:
                # Checking for co-occurrent connections to add to network
                if distance <= self.cooc_depth and word.upper() in self.cooc_dict:
                    cooc_words = list(self.cooc_dict[word.upper()])
                else:
                    cooc_words = []
                # Checking for semantic relations to add (only in network for now)
                if word.lower() in self.semantic_relations_dict:
                    assocs = [assoc.upper() for assoc in self.semantic_relations_dict[word.lower()]]
                    self.network.store(mem_id=word.upper(),
                                       time=1,
                                       activate=False,
                                       assocs=assocs,
                                       cooc_words=cooc_words)
                    next_sem_relations = assocs
                    next_words.update(cooc_words)
                    next_words.update(next_sem_relations)
                else:
                    self.network.store(mem_id=word.upper(),
                                       time=1,
                                       cooc_words=cooc_words,
                                       activate=False)
                    next_words.update(cooc_words)
            prev_words.update(curr_words)
            # Making sure we don't redundantly add any words
            curr_words = next_words - prev_words
            next_words = set()

    def create_blank_network(self):
        if self.bounded:
            self.network = sentenceLTM(
                activation_cls=(lambda ltm:
                                BoundedActivation(
                                    ltm,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))
        else:
            self.network = sentenceLTM(
                activation_cls=(lambda ltm:
                                SentenceCooccurrenceActivation(
                                    ltm,
                                    activation_base=self.activation_base,
                                    constant_offset=self.constant_offset,
                                    decay_parameter=self.decay_parameter
                                )))

    def clear_network(self, start_time=1):
        """
            Clears the semantic network by resetting activations to a certain "starting time".
            Parameters:
                sem_network (sentenceLTM): Network to clear.
                start_time (int): The network will be reset so that activations only at the starting time and before the
                    starting time remain.
            Returns:
                sentenceLTM: Cleared semantic network.
            """
        activations = self.network.activation.activations
        if start_time > 0:
            activated_words = activations.keys()
            for word in activated_words:
                activations[word] = [act for act in activations[word] if act[0] <= start_time]
            self.network.activation.activations = activations
        elif start_time == 0:
            self.network.activation.activations = defaultdict(list)
        else:
            raise ValueError(start_time)

    def get_distribution(self, context1, context2, context3):
        dist = {}
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        self.create_trial_network(context_list)
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth, activate=True)
        elements = sorted(set(self.network.activation.activations.keys()))
        for elem in elements:
            if elem in context_list:
                continue
            dist[elem] = self.network.get_activation(mem_id=elem, time=3)
        return dist

    def guess(self, context1, context2, context3):
        if self.spreading:
            spread_depth = -1
        else:
            spread_depth = 0
        context_list = [context1.upper(), context2.upper(), context3.upper()]
        self.create_trial_network(context_list)
        for context in context_list:
            self.network.store(mem_id=context.upper(), time=2, spread_depth=spread_depth, activate=True)
        max_act = -float("inf")
        guesses = []
        elements = sorted(set(self.network.activation.activations.keys()))
        for elem in elements:
            if elem in context_list:
                continue
            elem_act = self.network.get_activation(mem_id=elem, time=3)
            if elem_act is None:
                continue
            elif elem_act > max_act:
                max_act = elem_act
                guesses = [elem]
            elif elem_act == max_act:
                guesses.append(elem)
        return guesses


class AgentSpreadingBoostedCooc(AgentCooccurrence):
    # Adjusting cooc probabilities based on semantic activation...
    def __init__(self, ngrams=GoogleNGram('~/ngram'), source = 'SFFAN', func='sqrt'):
        super().__init__(ngrams)
        self.source = source
        self.func = func
        self.semantic_relations_dict = self.get_sem_rel_dict()
        self.cooc_dict = self.get_pairwise_cooc_dict()

    def to_string_id(self):
        result = "AgentSpreadingBoostedCooc"
        result += '_func_' + self.func
        result += '_source_' + self.source
        return result

    def get_pairwise_cooc_dict(self):
        cooc_cache = json.load(open("./n_gram_cooccurrence/single_ngrams.json"))
        cooc_cache_dict = defaultdict(list)
        for sublist in cooc_cache:
            for elem in sublist:
                cooc_cache_dict[tuple([elem[0][0].upper(), elem[0][1].upper()])] = elem[1]
        return cooc_cache_dict

    def get_sem_rel_dict(self):
        """ Gets dictionary of semantic relations from file."""
        sem_rel_link = "./semantic_relations_lists/" + self.source + "_sem_rel_dict.json"
        if os.path.isfile(sem_rel_link):
            sem_rel_file = open(sem_rel_link, "r")
            sem_rel_dict = json.load(sem_rel_file)
        else:
            raise ValueError()
        return sem_rel_dict

    def filter_sem_rel_dict(self, sem_rel_dict):
        """
        Filters the sem rel dict for stopwords to ensure that all words are valid.
        Parameters:
            sem_rel_dict (dictionary): A dictionary containing all semantic relations (the values) for each word
                (the keys) from the SWOWEN, SFFAN, (or both) databases.
        Returns:
            (dict) filtered semantic relations dictionary.
        """
        keys = sem_rel_dict.keys()
        for key in keys:
            rels = sem_rel_dict[key]
            words = [word for word in rels if word.lower() not in self.stopwords]
            sem_rel_dict[key] = words
        return sem_rel_dict

    def change_probability(self, probability):
        if self.func == 'sqrt':
            return math.sqrt(probability)
        elif self.func == 'sigmoid':
            return 1 / (1 + math.exp(-probability))
        elif self.func == 'cuberoot':
            return probability**(1/3)
        elif self.func == 'piecewise_sqrt':
            if probability <= 0.2:
                return 0.2
            elif probability > 0.2:
                return math.sqrt(probability)

    def find_probability(self, target, base):
        if tuple([target, base]) in self.cooc_dict:
            cooc_prob = self.cooc_dict[tuple([target, base])]
        elif tuple([base, target]) in self.cooc_dict:
            cooc_prob = self.cooc_dict[tuple([base, target])]
        else:
            cooc_prob = 0
        if target in self.semantic_relations_dict:
            rels = self.semantic_relations_dict[target]
        elif base in self.semantic_relations_dict:
            rels = self.semantic_relations_dict[base]
        else:
            return cooc_prob
        flag = False
        for rel in rels:
            if base in rels[rel]:
                flag = True
                break
        if flag:
            return self.change_probability(cooc_prob)
        else:
            return cooc_prob

    def get_distribution(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        dist = {}
        joint_cooc_set = self.cooc_cache[tuple([context1, context2, context3])]
        for elem in list(joint_cooc_set):
            cooc_word = elem[0]
            if cooc_word.lower() in self.stopwords:
                continue
            dist[cooc_word] = self.find_probability(cooc_word.upper(), context1) * self.find_probability(cooc_word.upper(), context2) * self.find_probability(cooc_word.upper(), context3)
        return dist

    def guess(self, context1, context2, context3):
        """
        Completes one trial of the RAT.
        Parameters:
            context1, context2, context3 (string): Context words to be used in the RAT task.
        Returns:
            A list of RAT guesses. Returns [] if there are no viable guesses.
        """
        joint_cooc_set = self.cooc_cache[tuple([context1, context2, context3])]
        if len(joint_cooc_set) == 0:
            return []
        elif len(joint_cooc_set) == 1:
            return list(joint_cooc_set)[0][0]
        else:
            max_cond_prob = -float("inf")
            max_elems = []
            for elem in list(joint_cooc_set):
                cooc_word = elem[0]
                if cooc_word.lower() in self.stopwords:
                    continue
                joint_cond_prob = self.find_probability(cooc_word.upper(), context1) * self.find_probability(cooc_word.upper(), context2) * self.find_probability(cooc_word.upper(), context3)
                if joint_cond_prob > max_cond_prob:
                    max_cond_prob = joint_cond_prob
                    max_elems = [cooc_word]
                elif joint_cond_prob == max_cond_prob:
                    max_elems.append(cooc_word)
            return max_elems


class AgentOracle(RATAgentTemplate):
    def __init__(self, activation_base=2, decay_parameter=0.05, constant_offset=0, ngrams=GoogleNGram('~/ngram')):
        self.spreading_sffan_agent = AgentSpreading(source="SFFAN", spreading=True, activation_base=activation_base,
                                                    decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                    bounded=False)
        self.spreading_sffan_bounded_agent = AgentSpreading(source="SFFAN", spreading=True, activation_base=activation_base,
                                                    decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                    bounded=True)
        self.cooccurrence_agent = AgentCooccurrence(ngrams=ngrams)

    def to_string_id(self):
        return "AgentOracle"

    def new_trial_callback(self):
        self.spreading_sffan_agent.clear_network(start_time=0)
        self.spreading_sffan_bounded_agent.clear_network(start_time=0)

    def guess_oracle(self, context1, context2, context3, answer):
        for agent in [self.cooccurrence_agent, self.spreading_sffan_agent, self.spreading_sffan_bounded_agent]:
            guess = agent.guess(context1, context2, context3)
            #print("guess", guess)
            #print("answer")
            if answer in guess:
                return [answer]
        return []


class RatTest:
    def __init__(self):
        self.rat_list = self.get_rat_list()

    def to_string_id(self):
        result = 'RAT'
        return result

    def get_rat_list(self):
        rat_file = csv.reader(open('./RAT/RAT_items.txt'))
        next(rat_file)
        return rat_file

    def is_correct(self, target, guesses):
        """Determines whether guesses to the sense of the target word are correct (True) or incorrect (False)"""
        correct_list = []
        for guess in guesses:
            if guess.upper() == target.upper():
                correct_list.append(True)
            else:
                correct_list.append(False)
        return correct_list

    def save_results(self, agent, accuracy_list):
        filename = self.to_string_id() + '_' + agent.to_string_id()
        with open('results/' + filename + ".json", 'w') as fd:
            json.dump(accuracy_list, fd)
            fd.close()

    def save_distribution(self, agent, avg_dist):
        #print(avg_dist)
        avg_dist_list = []
        for trial in avg_dist:
            target_word = trial[0]
            trial_candidates = trial[1]
            sublist = []
            if trial_candidates is None:
                continue
            for item in trial_candidates:
                sublist.append([item[0], item[1]])
            avg_dist_list.append([target_word, sublist])
        filename = self.to_string_id() + '_' + agent.to_string_id()
        with open('agent_distributions/list_' + filename + ".json", "w") as fd:
            json.dump(avg_dist_list, fd)
            fd.close()

    def run(self, agent):
        # initialize the agent (e.g. pre-activate concepts)
        agent.setup_run()
        # collect results
        accuracy_list = []
        counter = 0
        for trial in self.rat_list:
            counter += 1
            #if counter % 10 == 0:
                #print("trial", counter)
            agent.new_trial_callback()
            context1 = trial[0].upper()
            context2 = trial[1].upper()
            context3 = trial[2].upper()
            answer = trial[3].upper()
            if agent.to_string_id() == "AgentOracle":
                guesses = agent.guess_oracle(context1, context2, context3, answer)
            else:
                guesses = agent.guess(context1, context2, context3)
            correct = self.is_correct(answer, guesses)
            accuracy_list.append([guesses, correct])
            #print("solution:", answer, "guesses:", guesses)
        # save results to file
        self.save_results(agent, accuracy_list)
        #print("done!")
        return accuracy_list

    def get_rat_agent_distribution(self, agent):
        print(agent.to_string_id())
        # initialize the agent (e.g. pre-activate concepts)
        agent.setup_run()
        # collect distributions in nested dictionary
        dist = []
        counter = 0
        for trial in self.rat_list:
            counter += 1
            print(counter)
            #if counter % 10 == 0:
                #print("trial", counter)
            agent.new_trial_callback()
            context1 = trial[0].upper()
            context2 = trial[1].upper()
            context3 = trial[2].upper()
            answer = trial[3].upper()
            trial_dist_dict = agent.get_distribution(context1, context2, context3)
            trial_dist_list = []
            for elem in trial_dist_dict:
                trial_dist_list.append([elem, trial_dist_dict[elem]])
            dist.append([answer, trial_dist_list])
        # save results to file
        self.save_distribution(agent, dist)
        return dist


def calc_unifrandom():
    cooc_cache = json.load(open("./n_gram_cooccurrence/ngrams_cooccurrence_cache.json"))
    sum = 0
    count = 0
    for entry in cooc_cache:
        count += 1
        if len(entry[1]) != 0:
            sum += 1/len(entry[1])
    return sum / count



def create_RAT_agent(guess_method, source='SFFAN', spreading=True, activation_base=2, decay_parameter=0.05,
                     constant_offset=0, ngrams=GoogleNGram('~/ngram'), threshold=0, var_type='stdev', discount=0.1,
                     bounded=False, cooc_depth=1, func='sqrt'):
    if guess_method == 'spreading':
        return AgentSpreading(source=source, spreading=spreading, activation_base=activation_base,
                              decay_parameter=decay_parameter, constant_offset=constant_offset, bounded=bounded)
    elif guess_method == 'cooccurrence':
        return AgentCooccurrence(ngrams=ngrams)
    elif guess_method == 'cooc_thresh_sem':
        return AgentCoocThreshSpreading(source, spreading,
                                        activation_base, decay_parameter, constant_offset, ngrams, threshold,
                                        bounded=bounded)
    elif guess_method == 'sem_thresh_cooc':
        return AgentSpreadingThreshCooc(source, activation_base, decay_parameter, constant_offset, ngrams)
    elif guess_method == 'joint_prob':
        return AgentJointProbability(source, spreading, activation_base, decay_parameter, constant_offset, ngrams,
                                     bounded=bounded)
    elif guess_method == 'add_prob':
        return AgentAdditiveProbability(source, spreading, activation_base, decay_parameter, constant_offset, ngrams,
                                        bounded=bounded)
    elif guess_method == 'joint_var':
        return AgentJointVariance(source, spreading,
                                  activation_base, decay_parameter, constant_offset, ngrams, var_type, bounded=bounded)
    elif guess_method == 'max_prob':
        return AgentMaxProbability(source, spreading, activation_base, decay_parameter, constant_offset, ngrams,
                                   bounded=bounded)
    elif guess_method == 'oracle':
        return AgentOracle(activation_base, decay_parameter, constant_offset, ngrams)
    elif guess_method == 'cooc_weight_spreading':
        return AgentCoocWeightedSpreading(source, spreading, activation_base, decay_parameter, constant_offset,
                                          bounded=bounded)
    elif guess_method == 'bounded_spreading':
        return AgentBoundedSpreading(source=source, spreading=spreading, activation_base=activation_base,
                                     decay_parameter=decay_parameter, constant_offset=constant_offset)
    elif guess_method == 'cooc_supplemented_spreading':
        return AgentCoocSupplementedSpreading(source, discount, spreading, activation_base, decay_parameter,
                                              constant_offset, bounded=bounded)
    elif guess_method == 'spread_supplemented_cooc':
        return AgentSpreadingSupplementedCooc(source=source, activation_base=activation_base, decay_parameter=decay_parameter,
                                              constant_offset=constant_offset, ngrams=ngrams)
    elif guess_method == 'cooc_expanded_spreading':
        return AgentCoocExpandedSpreading(ngrams=ngrams, source=source, spreading=spreading, activation_base=activation_base,
                 decay_parameter=decay_parameter, constant_offset=constant_offset, bounded=bounded, cooc_depth=cooc_depth,
                                          threshold=threshold)
    elif guess_method == 'spreading_boosted_cooc':
        return AgentSpreadingBoostedCooc(ngrams=ngrams, source=source, func=func)

def analyze_data(results):
    denom = len(results)
    numer = 0
    for result in results:
        if len(result[1]) == 1 and result[1] == [True]:
            numer += 1
        elif len(result[1]) > 1 and True in result[1]:
            numer += 1 / len(result[1])
    return numer / denom


def main1():
    for guess_method in ['oracle']:
        rat = RatTest()
        # create agent
        agent = create_RAT_agent(guess_method=guess_method)
        # run (and saving data to disk)
        results = rat.run(agent)
        #print("results: ", results)
        # analyze
        #print("weighted avg: ", analyze_data(results))


def main_dists():
    for guess_method in ['spreading']:
        for partition in [1, 2, 3, 4, 5, 6]:
            for bounded in [True, False]:
                for clear in ['never', 'word', 'sentence']:
                    # create WSD task
                    rat = RatTest()
                    # create agent
                    agent = create_RAT_agent(guess_method=guess_method)
                    # run (and saving data to disk)
                    dists = rat.get_average_distribution(agent)
                    # analyze
