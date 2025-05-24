from collections import defaultdict
from sentence_long_term_memory import sentenceLTM
from sentence_cooccurrence_activation import *
import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn_corpus
import json
import math
import os.path
import numpy
import matplotlib.pyplot as plt


class WSDAgentTemplate:

    def to_string_id(self):
        result = ''
        # result += '_' + str(self.blah)
        # and so on
        return result

    def initialize(self):
        pass

    def new_sentence_callback(self):
        pass

    def new_word_callback(self):
        pass

    def guess(self, word_index, sentence):
        pass

    def setup_run(self):
        # Will be task dependent.
        pass


class AgentSpreading(WSDAgentTemplate):
    def __init__(self, partition, num_sentences, spreading, clear, activation_base=2, decay_parameter=0.05,
                 constant_offset=0, activate_answer=False, activate_sentence_words=False, bounded=False,
                 num_context_acts=1):
        self.partition = partition
        self.num_sentences = num_sentences
        self.corpus_utilities = CorpusUtilities(partition=partition, num_sentences=num_sentences)
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.spreading = spreading
        self.clear = clear
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.activate_answer = activate_answer
        self.activate_sentence_words = activate_sentence_words
        self.bounded = bounded
        self.time = 1
        self.network = self.create_sem_network()
        self.num_context_acts = num_context_acts

    def new_sentence_callback(self):
        if self.clear == 'sentence':
            self.clear_network(start_time=1)

    def new_word_callback(self):
        if self.clear == 'word':
            self.clear_network(start_time=1)

    def to_string_id(self):
        result = 'AgentSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + str(self.clear)
        result += '_depth_' + str(self.spreading)
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        result += '_context_acts' + str(self.num_context_acts)
        return result

    def get_semantic_relations_dict(self):
        """
            Gets the words related to each word in sentence_list and builds a dictionary to make the semantic network
            Parameters:
                sentence_list (list): list of all sentences or a partition of n sentences in the corpus
                partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                    at sentences 10000 - 14999.
                outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
                    relations are only considered from words inside the corpus.
            Returns:
                (dict) A dictionary with the semantic relations for every unique word in sentence_list
        """
        sem_rel_path = "./semantic_relations_lists/semantic_relations_list_inside_corpus"
        if len(self.sentence_list) == 30195:
            sem_rel_path = sem_rel_path + ".json"
        elif self.partition == 1:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + ".json"
        else:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + "_partition_" + str(
                self.partition) + ".json"
        if not os.path.isfile(sem_rel_path):
            semantic_relations_list = []
            # These are all the words in the corpus.
            semcor_words = set(sum(self.sentence_list, []))
            counter = 0
            for word in semcor_words:
                counter += 1
                syn = wn_corpus.synset(word[1])
                synonyms = [self.corpus_utilities.lemma_to_tuple(synon) for synon in syn.lemmas() if
                            self.corpus_utilities.lemma_to_tuple(synon) != word]
                # These are all synsets.
                synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                    syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                    syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                    syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                                    syn.verb_groups(), syn.similar_tos()]
                lemma_relations = []
                for relation in range(len(synset_relations)):
                    lemma_relations.append([])
                    # Getting each of the synsets above in synset_relations.
                    for syn in range(len(synset_relations[relation])):
                        # Getting the lemmas in each of the synset_relations synsets.
                        syn_lemmas = synset_relations[relation][syn].lemmas()
                        # Checking that lemmas from relations are in the corpus if outside_corpus=False
                        syn_lemmas = [lemma for lemma in syn_lemmas if lemma in semcor_words]
                        # Adding each lemma to the list
                        for syn_lemma in syn_lemmas:
                            lemma_tuple = self.corpus_utilities.lemma_to_tuple(syn_lemma)
                            if word != lemma_tuple:
                                lemma_relations[relation].append(lemma_tuple)
                word_sem_rel_subdict = self.create_word_sem_rel_dict(synonyms=synonyms,
                                                                     hypernyms=lemma_relations[0],
                                                                     hyponyms=lemma_relations[1],
                                                                     holonyms=lemma_relations[2],
                                                                     meronyms=lemma_relations[3],
                                                                     attributes=lemma_relations[4],
                                                                     entailments=lemma_relations[5],
                                                                     causes=lemma_relations[6],
                                                                     also_sees=lemma_relations[7],
                                                                     verb_groups=lemma_relations[8],
                                                                     similar_tos=lemma_relations[9])
                # Adding pairs of word & the dictionary containing its relations to the big json list (since json doesn't let lists be keys)
                # But we can still keep the word_sem_rel_subdict intact since its keys are strings
                semantic_relations_list.append([word, word_sem_rel_subdict])
            sem_rel_file = open(sem_rel_path, 'w')
            json.dump(semantic_relations_list, sem_rel_file)
            sem_rel_file.close()
        semantic_relations_list = json.load(open(sem_rel_path))
        semantic_relations_dict = {}
        for pair in semantic_relations_list:
            key = tuple(pair[0])
            val_dict = pair[1]
            for val_key in ["synonyms", "hypernyms", "hyponyms", "holonyms", "meronyms", "attributes",
                            "entailments",
                            "causes", "also_sees", "verb_groups", "similar_tos"]:
                list_val_vals = val_dict[val_key]
                tuple_val_vals = []
                for val_val in list_val_vals:
                    tuple_val_vals.append(tuple(val_val))
                val_dict[val_key] = tuple_val_vals
            semantic_relations_dict[key] = val_dict
        return semantic_relations_dict

    def create_word_sem_rel_dict(self, synonyms, hypernyms, hyponyms, holonyms, meronyms, attributes,
                                 entailments, causes, also_sees, verb_groups, similar_tos):
        """
        Creates a semantic relations dictionary with given semantic relations for a word.
        Parameters:
            synonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hypernyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hyponyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            holonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            meronyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            attributes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            entailments (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            causes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            also_sees (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            verb_groups (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            similar_tos (list) A list of word relations drawn from the synset a word belongs to from the nltk package
        Returns: A dictionary with the semantic relations for one word in the corpus.
        """
        sem_rel_dict = {"synonyms": set(synonyms), "hypernyms": set(hypernyms), "hyponyms": set(hyponyms),
                        "holonyms": set(holonyms), "meronyms": set(meronyms), "attributes": set(attributes),
                        "entailments": set(entailments), "causes": set(causes), "also_sees": set(also_sees),
                        "verb_groups": set(verb_groups), "similar_tos": set(similar_tos)}
        for rel in sem_rel_dict.keys():
            vals = sem_rel_dict[rel]
            string_vals = []
            for val in vals:
                string_vals.append(list(val))
            sem_rel_dict[rel] = string_vals
        return sem_rel_dict

    def create_sem_network(self):
        """
        Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
            hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
             Note that all words are stored at time 1.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict()
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
        relations_keys = list(semantic_relations_dict.keys())
        for word_index in range(len(relations_keys)):
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            network.store(mem_id=word_key,
                          time=self.time,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'])
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
        self.time = start_time
        if start_time > 0:
            activated_words = activations.keys()
            for word in activated_words:
                activations[word] = [act for act in activations[word] if act[0] <= start_time]
            self.network.activation.activations = activations
        elif start_time == 0:
            self.network.activation.activations = defaultdict(list)
        else:
            raise ValueError(start_time)

    def get_distribution(self, word_index, sentence):
        """
        """
        dist = {}
        self.time += 1
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        if not self.spreading:
            spread_depth = 0
        word = sentence[word_index]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for i in range(self.num_context_acts):
                for index in range(len(sentence)):
                    if index != word_index:
                        self.network.store(mem_id=sentence[index], time=self.time, spread_depth=spread_depth)
            self.time += 1
        other_senses = self.word_sense_dict[word[0]]
        # Getting the distribution
        for candidate in other_senses:
            candidate_act = self.network.get_activation(mem_id=candidate, time=self.time)
            dist[candidate] = candidate_act
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        if self.activate_answer:
            self.network.store(mem_id=word, time=self.time, spread_depth=spread_depth)
        for elem in max_guess:
            self.network.store(mem_id=elem, time=self.time, spread_depth=spread_depth)
        return dist

    def guess(self, word_index, sentence):
        """
                Gets guesses for a trial of the WSD.
                Parameters:
                    word (sense-word tuple): The word to guess the sense of, the "target word" (should not have sense-identifying
                        information).
                    context (list): A list of all possible senses of the target word, often obtained from the word sense
                        dictionary.
                    time (int): The time to calculate activations at.
                    network (sentenceLTM): Semantic network
                """
        self.time += 1
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        word = sentence[word_index]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for i in range(self.num_context_acts):
                for index in range(len(sentence)):
                    if index != word_index:
                        self.network.store(mem_id=sentence[index], time=self.time, spread_depth=spread_depth)
            self.time += 1
        other_senses = self.word_sense_dict[word[0]]
        for candidate in other_senses:
            candidate_act = self.network.get_activation(mem_id=candidate, time=self.time)
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        if self.activate_answer:
            self.network.store(mem_id=word, time=self.time, spread_depth=spread_depth)
        for elem in max_guess:
            self.network.store(mem_id=elem, time=self.time, spread_depth=spread_depth)
        return max_guess


class AgentCooccurrence(WSDAgentTemplate):
    def __init__(self, partition, num_sentences, context_type):
        self.partition = partition
        self.num_sentences = num_sentences
        self.context_type = context_type
        self.corpus_utilities = CorpusUtilities(partition=partition, num_sentences=num_sentences)
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.sense_counts = self.corpus_utilities.get_sense_counts()
        self.word_counts = self.corpus_utilities.get_word_counts()
        self.sense_sense_cooccurrences = self.corpus_utilities.get_sense_sense_cooccurrences()
        self.sense_word_cooccurrences = self.corpus_utilities.get_sense_word_cooccurrences()
        self.word_word_cooccurrences = self.corpus_utilities.get_word_word_cooccurrences()

    def to_string_id(self):
        result = 'AgentCooccurrence'
        result += '_context_' + self.context_type
        return result

    def get_count(self, *events):
        """
        Gets the counts of a single returned element, or two different elements for computing the conditional
        probability.
        Parameters:
            events (list): events to get the counts of.
        Returns:
            (int) counts of the events
        """
        if len(events) == 0:
            raise ValueError(events)
        if len(events) == 1:
            event = events[0]
            if type(event) == tuple:  # Context type = sense
                return self.sense_counts[event]
            else:  # Context type = word
                return self.word_counts[event]
        elif len(events) == 2:
            event1 = events[0]
            event2 = events[1]
            if type(event1) == tuple and type(event2) == tuple:  # Two senses
                if (event1, event2) not in self.sense_sense_cooccurrences.keys():
                    return 0
                return self.sense_sense_cooccurrences[(event1, event2)]
            elif type(event1) == tuple and type(event2) != tuple:  # event1 is a sense, event2 is a word
                if (event1, event2) not in self.sense_word_cooccurrences.keys():
                    return 0
                return self.sense_word_cooccurrences[(event1, event2)]
            elif type(event1) != tuple and type(event2) == tuple:  # event1 is a word, event2 is a sense
                if (event2, event1) not in self.sense_word_cooccurrences.keys():
                    return 0
                return self.sense_word_cooccurrences[(event2, event1)]
            else:  # event1 and event2 are words
                if (event1, event2) not in self.word_word_cooccurrences.keys():
                    return 0
                return self.word_word_cooccurrences[(event1, event2)]
        else:
            raise ValueError(events)

    def get_conditional_probability(self, target, base):
        """
        Gets conditional probability
        Parameters:
            target (tuple): Word of interest. Assumes that target is a sense (aka is formatted as a (word, sense) tuple)
            base (tuple or string): Context, can be a sense (described above) or a word (just a string, no sense
                information).
        Returns:
            (float) Decimal conditional probability
        """
        joint_count = self.get_count(base, target)  # How many times target SENSE & context cooccur
        base_count = self.get_count(base)  # How many times target context occurs
        if base_count == 0:
            return 0
        return joint_count / base_count

    def get_distribution(self, word_index, sentence):
        target_sense = sentence[word_index]
        dist = {}
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            for context_index in range(len(sentence)):
                if context_index != word_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.get_conditional_probability(target=target_sense_candidate,
                                                                                          base=context_word))
            dist[target_sense_candidate] = candidate_conditional_probability
        return dist

    def guess(self, word_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
        max_score = -float("inf")
        max_senses = None
        target_sense = sentence[word_index]
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            for context_index in range(len(sentence)):
                if context_index != word_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.get_conditional_probability(target=target_sense_candidate,
                                                                                          base=context_word))
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        return max_senses


class AgentCoocThreshSpreading(AgentSpreading):
    def __init__(self, partition, num_sentences, context_type, whole_corpus=False, spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0, activate_answer=False,
                 activate_sentence_words=False, bounded=False, threshold=0.0):
        """
                Parameters:
                    context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of the
                        context words ("sense") or not ("word")
                    whole_corpus (bool): For cooccurrence dependent corpus mechanisms, whether to include cooccurrent relations from
                        the whole corpus (True) or not (False).
                    corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                        Semcor corpus used
                    outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
                    activation_base (float): A parameter in the activation equation.
                    decay_parameter (float): A parameter in the activation equation.
                    constant_offset (float): A parameter in the activation equation.
                    spreading (bool): Whether to include the effects of spreading in the semantic network.
                """
        self.context_type = context_type
        self.threshold = threshold
        self.whole_corpus = whole_corpus
        self.agent_cooccurrence = AgentCooccurrence(num_sentences, partition, context_type)
        super().__init__(partition, num_sentences, spreading, clear, activation_base, decay_parameter, constant_offset,
                         activate_answer, activate_sentence_words, bounded)


    def to_string_id(self):
        result = 'AgentCoocThreshSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + str(self.clear)
        result += '_depth_' + str(self.spreading)
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        result += '_context_' + self.context_type
        result += '_whole_corpus_' + str(self.whole_corpus)
        return result

    def adjust_sem_rel_dict(self):
        """
        Adjusts the semantic relations dictionary to only include words that also cooccur with each word (as key).
        Parameters:
            sem_rel_dict (dict): A nested dictionary with each sense-specific word the key, and values the different
                semantic categories (synonyms, hyponyms, etc.) that the various sense-specific semantically related
                words are included in.
        Returns:
            (dict) Cooccurrence adjusted semantic relations dictionary.
        """
        sem_rel_path = "./semantic_relations_lists/semantic_relations_list"
        sem_rel_path = sem_rel_path + "_inside_corpus"
        if self.corpus_utilities.partition == 1:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list))
        else:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + "_partition_" + str(
                self.corpus_utilities.partition)
        sem_rel_path += "_thresh_" + str(self.threshold) + "_whole_corpus_" + str(self.whole_corpus) + ".json"
        if not os.path.isfile(sem_rel_path):
            sem_rel_dict = self.get_semantic_relations_dict()
            sent_list = self.agent_cooccurrence.corpus_utilities.get_sentence_list()
            sem_rel_list = []
            cooc_rel_dict = self.create_cooc_relations_dict(sent_list, context_type=self.context_type)
            for word_key in sem_rel_dict.keys():
                word_rel_dict = sem_rel_dict[word_key]  # has all different relations to target word
                for cat in word_rel_dict.keys():  # looping through each relation category
                    rels = word_rel_dict[cat]  # getting the relations in that category
                    new_rels = []
                    for rel in rels:  # going through words corresponding to each relation
                        if self.context_type == "sense":
                            base = rel
                        else:
                            base = rel[0]
                        if base in list(cooc_rel_dict[word_key]):
                            if self.threshold == 0:
                                new_rels.append(rel)
                            if self.agent_cooccurrence.get_conditional_probability(target=word_key,
                                                                                   base=base) > self.threshold:
                                new_rels.append(rel)
                    sem_rel_dict[word_key][cat] = new_rels
                sem_rel_list.append([word_key, sem_rel_dict[word_key]])
            sem_rel_file = open(sem_rel_path, 'w')
            json.dump(sem_rel_list, sem_rel_file)
            sem_rel_file.close()
            return sem_rel_dict
        sem_rel_list = json.load(open(sem_rel_path))
        sem_rel_dict = {}
        for pair in sem_rel_list:
            key = tuple(pair[0])
            val_dict = pair[1]
            for val_key in ["synonyms", "hypernyms", "hyponyms", "holonyms", "meronyms", "attributes",
                            "entailments",
                            "causes", "also_sees", "verb_groups", "similar_tos"]:
                list_val_vals = val_dict[val_key]
                tuple_val_vals = []
                for val_val in list_val_vals:
                    tuple_val_vals.append(tuple(val_val))
                val_dict[val_key] = tuple_val_vals
            sem_rel_dict[key] = val_dict
        return sem_rel_dict

    def create_cooc_relations_dict(self, sentence_list, context_type):
        """
        Creates a dictionary of cooccurrence relations for every word in the corpus.
        Parameters:
              sentence_list (nested list): A list of sentences from the Semcor corpus.
              context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense
                of the context words ("sense") or not ("word")
        Returns:
            (dict) Dictionary where each word in the corpus is a key and each of the other words it cooccurs with
                (are in the same sentence as our target word) are values (in a set).
        """
        cooc_rel_dict = defaultdict(set)
        for sent in sentence_list:
            for index in range(len(sent)):
                for context_index in range(len(sent)):
                    if index != context_index:
                        target_sense = sent[index]
                        context_sense = sent[context_index]
                        if context_type == "sense":
                            cooc_rel_dict[target_sense].update([context_sense])
                        else:
                            context_word = context_sense[0]
                            cooc_rel_dict[target_sense].update([context_word])
        #print(cooc_rel_dict)
        return cooc_rel_dict

    def create_sem_network(self):
        """ Builds corpus semantic network. """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.adjust_sem_rel_dict()
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
        relations_keys = sorted(list(set(semantic_relations_dict.keys())))
        for word_index in range(len(relations_keys)):
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            network.store(mem_id=word_key,
                          time=1,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'])
        return network


class AgentSpreadingThreshCooc(AgentCooccurrence):
    def __init__(self, partition, num_sentences, context_type):
        """
        Whole corpus is whether semantic relations should be required to cooccur with the target word over the whole
        corpus (True) or only in the partition of interest.
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                at sentences 10000 - 14999.
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used
            context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense of the
                context words ("sense") or not ("word")
            whole_corpus (bool): For cooccurrence dependent corpus mechanisms, whether to include cooccurrent relations from
                the whole corpus (True) or not (False).
        """
        super().__init__(partition, num_sentences, context_type)
        self.corpus_utilities = CorpusUtilities(num_sentences, partition)
        self.spreading_agent = AgentSpreading(partition, num_sentences, spreading=False, clear="never")
        self.sem_rel_dict = self.spreading_agent.get_semantic_relations_dict()
        if context_type == "word":
            self.sem_rel_dict = self.get_word_adjusted_sem_rel_dict(self.sem_rel_dict)
        self.sense_sense_cooccurrences = self.corpus_utilities.get_sense_sense_cooccurrences()
        self.sense_word_cooccurrences = self.corpus_utilities.get_sense_word_cooccurrences()
        self.word_word_cooccurrences = self.corpus_utilities.get_word_word_cooccurrences()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()

    def to_string_id(self):
        result = 'AgentSpreadingThreshCooc'
        result += '_context_' + self.context_type
        return result

    def get_word_adjusted_sem_rel_dict(self, sem_rel_dict):
        """
         Creates a word-based semantic relations dictionary (assuming we don't care about the sense of each
         semantically-related word).
         Parameters:
            sem_rel_dict (dict): A nested dictionary with each sense-specific word the key, and values the different
                semantic categories (synonyms, hyponyms, etc.) that the various sense-specific semantically related
                words are included in.
         Returns: (dict) Altered semantic relations dict that assumes only the sense of each semantically related word
            is not known.
        """
        keys = sem_rel_dict.keys()
        for word in keys:
            rels = sem_rel_dict[word]
            for rel in rels.keys():
                new_rel_list = []
                rel_words = rels[rel]
                for rel_word in rel_words:
                    new_rel_list.append(rel_word[0])
                sem_rel_dict[word][rel] = new_rel_list
        return sem_rel_dict

    def get_distribution(self, word_index, sentence):
        """
        """
        dist = {}
        target_sense = sentence[word_index]
        # for each retrieval candidate...
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            # for each context word...
            for context_index in range(len(sentence)):
                if context_index == word_index:
                    continue
                context_sense = sentence[context_index]
                if self.context_type == "sense":
                    base = context_sense
                    organized_target_rels = self.sem_rel_dict[target_sense_candidate]
                    target_rels = sum(list(organized_target_rels.values()), [])
                    if context_sense not in target_rels:
                        continue
                else:  # context == "word"
                    context_word = context_sense[0]
                    base = context_word
                    organized_target_rels = self.sem_rel_dict[target_sense_candidate]
                    target_rels = sum(list(organized_target_rels.values()), [])
                    if context_word not in target_rels:
                        continue
                candidate_conditional_probability = (candidate_conditional_probability *
                                                     self.get_conditional_probability(target=target_sense_candidate,
                                                                                      base=base))
            dist[target_sense_candidate] = candidate_conditional_probability
        return dist

    def guess(self, word_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
        max_score = -float("inf")
        max_senses = None
        target_sense = sentence[word_index]
        # for each retrieval candidate...
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            # for each context word...
            for context_index in range(len(sentence)):
                if context_index == word_index:
                    continue
                context_sense = sentence[context_index]
                if self.context_type == "sense":
                    base = context_sense
                    organized_target_rels = self.sem_rel_dict[target_sense_candidate]
                    target_rels = sum(list(organized_target_rels.values()), [])
                    if context_sense not in target_rels:
                        continue
                else:  # context == "word"
                    context_word = context_sense[0]
                    base = context_word
                    organized_target_rels = self.sem_rel_dict[target_sense_candidate]
                    target_rels = sum(list(organized_target_rels.values()), [])
                    if context_word not in target_rels:
                        continue
                candidate_conditional_probability = (candidate_conditional_probability *
                                                     self.get_conditional_probability(target=target_sense_candidate,
                                                                                      base=base))
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        if max_score == -float("inf") or max_score == 0:
            return []
        return max_senses


class AgentJointProbability(WSDAgentTemplate):
    def __init__(self, partition, num_sentences, context_type, spreading=True,
                 clear="never", activation_base=2.0, decay_parameter=0.05, constant_offset=0.0, activate_answer=False,
                 activate_sentence_words=False, bounded=False):
        self.partition = partition
        self.num_sentences = num_sentences
        self.context_type = context_type
        self.spreading = spreading
        self.clear = clear
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.activate_answer = activate_answer
        self.activate_sentence_words = activate_sentence_words
        self.bounded = bounded
        self.corpus_utilities = CorpusUtilities(num_sentences, partition)
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.cooc_agent = AgentCooccurrence(partition, num_sentences, context_type)
        self.spreading_agent = AgentSpreading(partition, num_sentences, spreading, clear, activation_base,
                                              decay_parameter, constant_offset, activate_answer=activate_answer,
                                              activate_sentence_words=activate_sentence_words, bounded=bounded)

    def to_string_id(self):
        result = 'AgentJointProbability'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + str(self.clear)
        result += '_depth_' + str(self.spreading)
        result += '_context_' + self.context_type
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        return result

    def new_sentence_callback(self):
        if self.clear == 'sentence':
            self.spreading_agent.clear_network(start_time=1)

    def new_word_callback(self):
        if self.clear == 'word':
            self.spreading_agent.clear_network(start_time=1)

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


    def get_conditional_probability(self, word, context):
        """ Gets conditional probability for word across whole sentence (context)"""
        conditional_prob = 1
        for con in context:
            if self.context_type == "word":
                conditional_prob = (conditional_prob *
                                    self.cooc_agent.get_conditional_probability(target=word, base=con[0]))
            else:
                conditional_prob = conditional_prob * self.cooc_agent.get_conditional_probability(target=word, base=con)
        return conditional_prob

    def get_cooccurrence_distribution(self, target_index, word_senses, sentence, context_type):
        """ Creates a dictionary of the conditional distribution for every possible sense of a certain word in the
         sentence given.
         Assumes word is not in the context. """
        conditional_probs = defaultdict(float)
        cooc_sum = 1
        for sense in word_senses:
            conditional_prob = 1
            for con_index in range(len(sentence)):
                if con_index == target_index:
                    continue
                con = sentence[con_index]
                if context_type == "word":
                    conditional_prob = (conditional_prob *
                                        self.cooc_agent.get_conditional_probability(target=sense, base=con[0]))
                else:
                    conditional_prob = (conditional_prob *
                                        self.cooc_agent.get_conditional_probability(target=sense, base=con))
            conditional_probs[sense] = conditional_prob
            cooc_sum += conditional_prob
        if cooc_sum != 0:
            for key in conditional_probs.keys():
                conditional_probs[key] = conditional_probs[key] / cooc_sum
        return conditional_probs

    def get_spreading_distribution(self, word_senses, time, tau=-float("inf"), s=0.25):
        """ Gets distribution of semantic activation values for all senses of a word given.
            Options for clear are "never", "sentence", "word" """
        sense_acts = defaultdict(float)
        for sense in word_senses:
            sense_acts[sense] = self.spreading_agent.network.get_activation(sense, time)
        sense_act_probs = defaultdict(float)
        for sense in word_senses:
            other_acts = list(sense_acts.values())
            sense_act_probs[sense] = self.get_activation_probability(word_activation=sense_acts[sense],
                                                                     other_activations=other_acts,
                                                                     tau=tau, s=s)
        return sense_act_probs

    def get_distribution(self, word_index, sentence):
        self.spreading_agent.time += 1
        word = sentence[word_index]
        senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.spreading_agent.network.store(mem_id=sentence[index], time=self.spreading_agent.time)
            self.spreading_agent.time += 1
        spread_dist = self.get_spreading_distribution(senses, self.spreading_agent.time)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=word_index, sentence=sentence)
        joint_probs = self.create_joint_distribution(spreading_dist=spread_dist, cooccurrence_dist=cooc_dist)
        max_joint = 0
        guesses = []
        for key in list(joint_probs.keys()):
            joint = joint_probs[key]
            if joint > max_joint:
                guesses = [key]
                max_joint = joint
            if joint == max_joint:
                guesses.append(key)
        for guess in guesses:
            self.spreading_agent.network.store(guess, self.spreading_agent.time)
        if self.activate_answer:
            self.spreading_agent.network.store(mem_id=word, time=self.spreading_agent.time)
        return joint_probs

    def guess(self, word_index, sentence):
        """ Does one trial of the WSD task."""
        self.spreading_agent.time += 1
        word = sentence[word_index]
        senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.spreading_agent.network.store(mem_id=sentence[index], time=self.spreading_agent.time)
            self.spreading_agent.time += 1
        spread_dist = self.get_spreading_distribution(senses, self.spreading_agent.time)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=word_index, sentence=sentence)
        joint_probs = self.create_joint_distribution(spreading_dist=spread_dist, cooccurrence_dist=cooc_dist)
        max_joint = 0
        guesses = []
        for key in list(joint_probs.keys()):
            joint = joint_probs[key]
            if joint > max_joint:
                guesses = [key]
                max_joint = joint
            elif joint == max_joint:
                guesses.append(key)
        for guess in guesses:
            self.spreading_agent.network.store(guess, self.spreading_agent.time)
        if self.activate_answer:
            self.spreading_agent.network.store(mem_id=word, time=self.spreading_agent.time)
        return guesses


class AgentAdditiveProbability(AgentJointProbability):
    def __init__(self, partition, num_sentences, context_type, spreading=True, clear="never", activation_base=2.0,
                 decay_parameter=0.05, constant_offset=0.0, activate_answer=False, activate_sentence_words=False,
                 bounded=False):
        super().__init__(partition, num_sentences, context_type, spreading, clear, activation_base, decay_parameter,
                         constant_offset, activate_answer, activate_sentence_words, bounded)

    def to_string_id(self):
        result = 'AgentAdditiveProbability'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + str(self.clear)
        result += '_depth_' + str(self.spreading)
        result += '_context_' + self.context_type
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
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


class AgentJointVariance(WSDAgentTemplate):
    def __init__(self, partition, num_sentences, context_type, clear,
                 activation_base=2.0, decay_parameter=0.05, constant_offset=0.0,
                 activate_answer=False, activate_sentence_words=False, bounded=False, var_type="stdev"):
        self.num_sentences = num_sentences
        self.partition = partition
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.activate_answer = activate_answer
        self.activate_sentence_words = activate_sentence_words
        self.bounded = bounded
        self.var_type = var_type
        self.corpus_utilities = CorpusUtilities(num_sentences, partition)
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.context_type = context_type
        self.clear = clear
        self.cooc_agent = AgentCooccurrence(self.partition, self.num_sentences, context_type=context_type)
        self.spreading_agent = AgentSpreading(self.partition, self.num_sentences, spreading=True, clear=clear,
                                              activation_base=activation_base, decay_parameter=decay_parameter,
                                              constant_offset=constant_offset, activate_answer=activate_answer,
                                              activate_sentence_words=activate_sentence_words, bounded=bounded)

    def to_string_id(self):
        result = 'AgentJointVariance'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + str(self.clear)
        result += '_context_' + self.context_type
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        result += '_var_' + self.var_type
        return result

    def new_sentence_callback(self):
        if self.clear == 'sentence':
            self.spreading_agent.clear_network(start_time=1)

    def new_word_callback(self):
        if self.clear == 'word':
            self.spreading_agent.clear_network(start_time=1)

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

    def get_activation_probability(self, word_activation, other_activations, tau, s=0.25):
        """ Gets activation probability for a given element with a specified activation """
        num = math.exp(word_activation / s)
        denom = math.exp(tau / s) + sum(math.exp(act / s) for act in other_activations)
        return num / denom

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

    def get_cooccurrence_distribution(self, target_index, word_senses, sentence, context_type):
        """ Creates a dictionary of the conditional distribution for every possible sense of a certain word in the
         sentence given.
         Assumes word is not in the context. """
        conditional_probs = defaultdict(float)
        cooc_sum = 0
        for sense in word_senses:
            conditional_prob = 1
            for con_index in range(len(sentence)):
                if con_index == target_index:
                    continue
                con = sentence[con_index]
                if context_type == "word":
                    conditional_prob = (conditional_prob *
                                        self.cooc_agent.get_conditional_probability(target=sense, base=con[0]))
                else:
                    conditional_prob = (conditional_prob *
                                        self.cooc_agent.get_conditional_probability(target=sense, base=con))
            conditional_probs[sense] = conditional_prob
            cooc_sum += conditional_prob
        for key in conditional_probs.keys():
            conditional_probs[key] = conditional_probs[key] / cooc_sum
        return conditional_probs

    def get_spreading_distribution(self, word_senses, time, tau=-float("inf"), s=0.25):
        """ Gets distribution of semantic activation values for all senses of a word given.
            Options for clear are "never", "sentence", "word" """
        sense_acts = defaultdict(float)
        for sense in word_senses:
            sense_acts[sense] = self.spreading_agent.network.get_activation(sense, time)
        sense_act_probs = defaultdict(float)
        for sense in word_senses:
            other_acts = list(sense_acts.values())
            sense_act_probs[sense] = self.get_activation_probability(word_activation=sense_acts[sense],
                                                                     other_activations=other_acts,
                                                                     tau=tau, s=s)
        return sense_act_probs

    def get_distribution(self, word_index, sentence):
        """ Does one trial of the WSD task."""
        self.spreading_agent.time += 1
        word = sentence[word_index]
        senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.spreading_agent.network.store(mem_id=sentence[index], time=self.spreading_agent.time)
            self.spreading_agent.time += 1
        spread_dist = self.get_spreading_distribution(senses, self.spreading_agent.time)
        spread_dist_var = self.get_distribution_variance(spread_dist)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=word_index, sentence=sentence)
        cooc_dist_var = self.get_distribution_variance(cooc_dist)
        if cooc_dist_var >= spread_dist_var:
            dist = cooc_dist
            guesses = self.get_guesses(cooc_dist)
        else:
            dist = spread_dist
            guesses = self.get_guesses(spread_dist)
        for guess in guesses:
            self.spreading_agent.network.store(guess, self.spreading_agent.time)
        if self.activate_answer:
            self.spreading_agent.network.store(mem_id=word, time=self.spreading_agent.time)
        return dist

    def guess(self, word_index, sentence):
        """ Does one trial of the WSD task."""
        self.spreading_agent.time += 1
        word = sentence[word_index]
        senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.spreading_agent.network.store(mem_id=sentence[index], time=self.spreading_agent.time)
            self.spreading_agent.time += 1
        spread_dist = self.get_spreading_distribution(senses, self.spreading_agent.time)
        spread_dist_var = self.get_distribution_variance(spread_dist)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=word_index, sentence=sentence)
        cooc_dist_var = self.get_distribution_variance(cooc_dist)
        if cooc_dist_var >= spread_dist_var:
            guesses = self.get_guesses(cooc_dist)
        else:
            guesses = self.get_guesses(spread_dist)
        for guess in guesses:
            self.spreading_agent.network.store(guess, self.spreading_agent.time)
        if self.activate_answer:
            self.spreading_agent.network.store(mem_id=word, time=self.spreading_agent.time)
        return guesses


class AgentMaxProbability(AgentJointProbability):

    def __init__(self, partition, num_sentences, context_type, spreading=True, clear="never", activation_base=2.0,
                 decay_parameter=0.05, constant_offset=0.0, activate_answer=False, activate_sentence_words=False,
                 bounded=False):
        super().__init__(partition, num_sentences, context_type, spreading, clear, activation_base, decay_parameter,
                         constant_offset, activate_answer, activate_sentence_words, bounded)

    def to_string_id(self):
        result = 'AgentMaxProbability'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + str(self.clear)
        result += '_depth_' + str(self.spreading)
        result += '_context_' + self.context_type
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        return result

    def get_distribution(self, word_index, sentence):
        """ Does one trial of the WSD task."""
        self.spreading_agent.time += 1
        word = sentence[word_index]
        senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.spreading_agent.network.store(mem_id=sentence[index], time=self.spreading_agent.time)
            self.spreading_agent.time += 1
        spread_dist = self.get_spreading_distribution(senses, self.spreading_agent.time)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=word_index, sentence=sentence)
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

        if not spread_dist:
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
        for guess in guesses:
            self.spreading_agent.network.store(guess, self.spreading_agent.time)
        if self.activate_answer:
            self.spreading_agent.network.store(mem_id=word, time=self.spreading_agent.time)
        return dist

    def guess(self, word_index, sentence):
        """ Does one trial of the WSD task."""
        self.spreading_agent.time += 1
        word = sentence[word_index]
        senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.spreading_agent.network.store(mem_id=sentence[index], time=self.spreading_agent.time)
            self.spreading_agent.time += 1
        spread_dist = self.get_spreading_distribution(senses, self.spreading_agent.time)
        cooc_dist = self.get_cooccurrence_distribution(context_type=self.context_type, word_senses=senses,
                                                       target_index=word_index, sentence=sentence)
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
        for guess in guesses:
            self.spreading_agent.network.store(guess, self.spreading_agent.time)
        if self.activate_answer:
            self.spreading_agent.network.store(mem_id=word, time=self.spreading_agent.time)
        return guesses


class AgentCoocWeightedSpreading(AgentSpreading):
    def __init__(self, partition, num_sentences, spreading, clear, context_type, activation_base=2,
                 decay_parameter=0.05, constant_offset=0, activate_answer=False, activate_sentence_words=False,
                 bounded=False):
        self.partition = partition
        self.num_sentences = num_sentences
        self.corpus_utilities = CorpusUtilities(partition=partition, num_sentences=num_sentences)
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.spreading = spreading
        self.clear = clear
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.context_type = context_type
        self.cooc_agent = AgentCooccurrence(partition, num_sentences, context_type)
        self.cooc_dict = self.create_cooc_dict()
        super().__init__(partition, num_sentences, spreading, clear, activation_base, decay_parameter, constant_offset,
                         activate_answer, activate_sentence_words, bounded)

    def to_string_id(self):
        result = 'AgentCooccurrenceWeightedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + self.clear
        result += '_depth_' + str(self.spreading)
        result += '_context_' + self.context_type
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        return result

    def create_cooc_dict(self):
        cooc_dict = defaultdict(float)
        semantic_relations_dict = self.get_semantic_relations_dict()
        for key in semantic_relations_dict.keys():
            rel_type_vals = semantic_relations_dict[key] # This is a dictionary.
            for rel_type_key in rel_type_vals.keys():
                vals = rel_type_vals[rel_type_key]
                if len(vals) == 0:
                    continue
                for val in vals:
                    cooc_dict[tuple([key, val])] = self.cooc_agent.get_conditional_probability(target=key, base=val)
        return cooc_dict

    def create_sem_network(self):
        """
        Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
            hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
             Note that all words are stored at time 1.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict()
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
        relations_keys = list(semantic_relations_dict.keys())
        for word_index in range(len(relations_keys)):
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            network.store(mem_id=word_key,
                          time=self.time,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'])
        return network


class AgentBoundedSpreading(AgentSpreading):

    def __init__(self, partition, num_sentences, spreading, clear, activation_base=2,
                 decay_parameter=0.05, constant_offset=0):
        super().__init__(partition, num_sentences, spreading, clear, activation_base, decay_parameter, constant_offset)

    def to_string_id(self):
        result = 'AgentBoundedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_clear_' + self.clear
        result += '_depth_' + str(self.spreading)
        return result

    def create_sem_network(self):
        """
        Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
            hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
             Note that all words are stored at time 1.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict()
        network = sentenceLTM(
            activation_cls=(lambda ltm:
                            BoundedActivation(
                                ltm,
                                activation_base=self.activation_base,
                                constant_offset=self.constant_offset,
                                decay_parameter=self.decay_parameter
                            )))
        relations_keys = list(semantic_relations_dict.keys())
        for word_index in range(len(relations_keys)):
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            network.store(mem_id=word_key,
                          time=self.time,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'])
        return network


class AgentSpreadingSupplementedCooc(AgentCooccurrence):
    def __init__(self, partition, num_sentences, context_type, discount=0.1):
        super().__init__(partition, num_sentences, context_type)
        self.discount = discount
        self.spreading_agent = AgentSpreading(partition, num_sentences, spreading=False, clear="never")
        self.sem_rel_dict = self.spreading_agent.get_semantic_relations_dict()
        if context_type == "word":
            self.sem_rel_dict = self.get_word_adjusted_sem_rel_dict(self.sem_rel_dict)

    def to_string_id(self):
        result = 'AgentCooccurrence'
        result += '_context_' + self.context_type
        result += '_discount_' + str(self.discount)
        return result

    def get_word_adjusted_sem_rel_dict(self, sem_rel_dict):
        """
         Creates a word-based semantic relations dictionary (assuming we don't care about the sense of each
         semantically-related word).
         Parameters:
            sem_rel_dict (dict): A nested dictionary with each sense-specific word the key, and values the different
                semantic categories (synonyms, hyponyms, etc.) that the various sense-specific semantically related
                words are included in.
         Returns: (dict) Altered semantic relations dict that assumes only the sense of each semantically related word
            is not known.
        """
        keys = sem_rel_dict.keys()
        for word in keys:
            rels = sem_rel_dict[word]
            for rel in rels.keys():
                new_rel_list = []
                rel_words = rels[rel]
                for rel_word in rel_words:
                    new_rel_list.append(rel_word[0])
                sem_rel_dict[word][rel] = new_rel_list
        return sem_rel_dict

    def get_associated_words(self, word):
        rels = []
        if word in self.sem_rel_dict:
            assocs = self.sem_rel_dict[word]
            for key in assocs.keys():
                vals = assocs[key]
                for val in vals:
                    if self.context_type == "word":
                        val = val[0]
                    rels.append(val)
        return rels

    def get_distribution(self, word_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
        dist = {}
        max_score = -float("inf")
        max_senses = []
        target_sense = sentence[word_index]
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            for context_index in range(len(sentence)):
                if context_index != word_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.get_conditional_probability(target_sense_candidate,
                                                                                          context_word))
            added_words = self.get_associated_words(target_sense_candidate)
            for context_word in added_words:
                if context_word not in sentence:
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.get_conditional_probability(target_sense_candidate,
                                                                                          context_word))
            dist[target_sense_candidate] = candidate_conditional_probability
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        return dist

    def guess(self, word_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
        max_score = -float("inf")
        max_senses = []
        target_sense = sentence[word_index]
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            for context_index in range(len(sentence)):
                if context_index != word_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.get_conditional_probability(target_sense_candidate,
                                                                                          context_word))
            added_words = self.get_associated_words(target_sense_candidate)
            for context_word in added_words:
                if context_word not in sentence:
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.get_conditional_probability(target_sense_candidate,
                                                                                          context_word))
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        return max_senses


class AgentCoocSupplementedSpreading(AgentSpreading):
    def __init__(self, partition, num_sentences, spreading, clear, activation_base=2, decay_parameter=0.05,
                 constant_offset=0, context_type="word", activate_answer=False, activate_sentence_words=False,
                 bounded=False, discount=0.1):
        self.context_type = context_type
        self.corpus_utilities = CorpusUtilities(partition=partition, num_sentences=num_sentences)
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.cooc_dict = self.create_cooc_relations_dict(self.sentence_list, self.context_type)
        self.discount = discount
        super().__init__(partition, num_sentences, spreading, clear, activation_base, decay_parameter, constant_offset,
                         activate_answer, activate_sentence_words, bounded)

    def to_string_id(self):
        result = 'AgentCoocSupplementedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        result += '_clear_' + self.clear
        result += '_depth_' + str(self.spreading)
        result += '_discount_' + str(self.discount)
        return result


    def create_sem_network(self):
        """
        Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
            hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
             Note that all words are stored at time 1.
        Returns:
            network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
        """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict()
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
        relations_keys = list(semantic_relations_dict.keys())
        for word_index in range(len(relations_keys)):
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            network.store(mem_id=word_key,
                          time=self.time,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'])
        return network


    def create_cooc_relations_dict(self, sentence_list, context_type):
        """
        Creates a dictionary of cooccurrence relations for every word in the corpus.
        Parameters:
              sentence_list (nested list): A list of sentences from the Semcor corpus.
              context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense
                of the context words ("sense") or not ("word")
        Returns:
            (dict) Dictionary where each word in the corpus is a key and each of the other words it cooccurs with
                (are in the same sentence as our target word) are values (in a set).
        """
        cooc_rel_dict = defaultdict(set)
        for sent in sentence_list:
            for index in range(len(sent)):
                for context_index in range(len(sent)):
                    if index != context_index:
                        target_sense = sent[index]
                        context_sense = sent[context_index]
                        if context_type == "sense":
                            cooc_rel_dict[target_sense[0]].update([context_sense])
                        else:
                            context_word = context_sense[0]
                            cooc_rel_dict[target_sense[0]].update([context_word])
        return cooc_rel_dict

    def get_additional_words(self, word):
        if word in self.cooc_dict:
            return self.cooc_dict[word]
        else:
            return []

    def get_distribution(self, word_index, sentence):
        """
                Gets guesses for a trial of the WSD.
                Parameters:
                    word (sense-word tuple): The word to guess the sense of, the "target word" (should not have sense-identifying
                        information).
                    context (list): A list of all possible senses of the target word, often obtained from the word sense
                        dictionary.
                    time (int): The time to calculate activations at.
                    network (sentenceLTM): Semantic network
                """
        dist = {}
        self.time += 1
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        word = sentence[word_index]
        other_senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.network.store(mem_id=sentence[index], time=self.time)
        # Now activating additional words as context...
        added_words = self.get_additional_words(word[0])
        for added_word in added_words:
            self.network.store(mem_id=added_word, time=self.time)
        self.time += 1
        for candidate in other_senses:
            candidate_act = math.exp(self.network.get_activation(mem_id=candidate, time=self.time))
            #added_words = self.get_additional_words(candidate)
            #for candidate_word in added_words:
            #    if self.network.get_activation(mem_id=candidate_word, time=self.time) is not None: # Don't know if this is the right thing to do...
            #        candidate_act = candidate_act * math.exp(self.network.get_activation(mem_id=candidate_word, time=self.time))
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
            dist[candidate] = candidate_act
        for elem in max_guess:
            self.network.store(mem_id=elem, time=self.time, spread_depth=spread_depth)
        if self.activate_answer:
            self.network.store(mem_id=word, time=self.time)
        return dist

    def guess(self, word_index, sentence):
        """
                Gets guesses for a trial of the WSD.
                Parameters:
                    word (sense-word tuple): The word to guess the sense of, the "target word" (should not have sense-identifying
                        information).
                    context (list): A list of all possible senses of the target word, often obtained from the word sense
                        dictionary.
                    time (int): The time to calculate activations at.
                    network (sentenceLTM): Semantic network
                """
        self.time += 1
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        word = sentence[word_index]
        other_senses = self.word_sense_dict[word[0]]
        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.network.store(mem_id=sentence[index], time=self.time)
        self.time += 1
        for candidate in other_senses:
            candidate_act = math.exp(self.network.get_activation(mem_id=candidate, time=self.time))
            added_words = self.get_additional_words(candidate)
            for candidate_word in added_words:
                if self.network.get_activation(mem_id=candidate_word, time=self.time) is not None: # Don't know if this is the right thing to do...
                    candidate_act = candidate_act * math.exp(self.network.get_activation(mem_id=candidate_word, time=self.time))
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        for elem in max_guess:
            self.network.store(mem_id=elem, time=self.time, spread_depth=spread_depth)
        if self.activate_answer:
            self.network.store(mem_id=word, time=self.time)
        return max_guess


class AgentCoocExpandedSpreading(WSDAgentTemplate):

    def __init__(self, partition, num_sentences, context_type, spreading, clear, activation_base=2,
                 decay_parameter=0.05, constant_offset=0, activate_answer=False, activate_sentence_words=False,
                 bounded=False, cooc_depth=1):
        self.context_type = context_type
        self.partition = partition
        self.num_sentences = num_sentences
        self.corpus_utilities = CorpusUtilities(partition=partition, num_sentences=num_sentences)
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.spreading = spreading
        self.clear = clear
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.activate_answer = activate_answer
        self.activate_sentence_words = activate_sentence_words
        self.bounded = bounded
        self.time = 0
        self.cooc_depth = cooc_depth # A cooc_depth of 1 means it only adds cooccurrent connections to all context words...
        self.semantic_relations_dict = self.get_semantic_relations_dict()
        self.cooc_dict = self.get_cooc_relations_dict()
        self.create_blank_network()


    def to_string_id(self):
        result = 'AgentCoocExpandedSpreading'
        result += '_' + str(self.activation_base)
        result += '_' + str(self.decay_parameter)
        result += '_' + str(self.constant_offset)
        result += 'context' + str(self.context_type)
        result += '_clear_' + self.clear
        result += '_depth_' + str(self.spreading)
        result += '_act_answer_' + str(self.activate_answer)
        result += '_act_sentence_words_' + str(self.activate_sentence_words)
        result += '_bounded_' + str(self.bounded)
        result += '_coocdepth_' + str(self.cooc_depth)
        return result

    def new_sentence_callback(self):
        if self.clear == 'sentence':
            # Restarting the network...
            self.create_blank_network()
            self.time = 0

    def new_word_callback(self):
        if self.clear == 'word':
            # Restarting the network...
            self.create_blank_network()
            self.time = 0

    def create_word_sem_rel_dict(self, synonyms, hypernyms, hyponyms, holonyms, meronyms, attributes,
                                 entailments, causes, also_sees, verb_groups, similar_tos):
        """
        Creates a semantic relations dictionary with given semantic relations for a word.
        Parameters:
            synonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hypernyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hyponyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            holonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            meronyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            attributes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            entailments (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            causes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            also_sees (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            verb_groups (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            similar_tos (list) A list of word relations drawn from the synset a word belongs to from the nltk package
        Returns: A dictionary with the semantic relations for one word in the corpus.
        """
        sem_rel_dict = {"synonyms": set(synonyms), "hypernyms": set(hypernyms), "hyponyms": set(hyponyms),
                        "holonyms": set(holonyms), "meronyms": set(meronyms), "attributes": set(attributes),
                        "entailments": set(entailments), "causes": set(causes), "also_sees": set(also_sees),
                        "verb_groups": set(verb_groups), "similar_tos": set(similar_tos)}
        for rel in sem_rel_dict.keys():
            vals = sem_rel_dict[rel]
            string_vals = []
            for val in vals:
                string_vals.append(list(val))
            sem_rel_dict[rel] = string_vals
        return sem_rel_dict


    def get_semantic_relations_dict(self):
        """
            Gets the words related to each word in sentence_list and builds a dictionary to make the semantic network
            Parameters:
                sentence_list (list): list of all sentences or a partition of n sentences in the corpus
                partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                    at sentences 10000 - 14999.
                outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
                    relations are only considered from words inside the corpus.
            Returns:
                (dict) A dictionary with the semantic relations for every unique word in sentence_list
        """
        sem_rel_path = "./semantic_relations_lists/semantic_relations_list_inside_corpus"
        if len(self.sentence_list) == 30195:
            sem_rel_path = sem_rel_path + ".json"
        elif self.partition == 1:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + ".json"
        else:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + "_partition_" + str(
                self.partition) + ".json"
        if not os.path.isfile(sem_rel_path):
            semantic_relations_list = []
            # These are all the words in the corpus.
            semcor_words = set(sum(self.sentence_list, []))
            counter = 0
            for word in semcor_words:
                counter += 1
                syn = wn_corpus.synset(word[1])
                synonyms = [self.corpus_utilities.lemma_to_tuple(synon) for synon in syn.lemmas() if
                            self.corpus_utilities.lemma_to_tuple(synon) != word]
                # These are all synsets.
                synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                    syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                    syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                    syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                                    syn.verb_groups(), syn.similar_tos()]
                lemma_relations = []
                for relation in range(len(synset_relations)):
                    lemma_relations.append([])
                    # Getting each of the synsets above in synset_relations.
                    for syn in range(len(synset_relations[relation])):
                        # Getting the lemmas in each of the synset_relations synsets.
                        syn_lemmas = synset_relations[relation][syn].lemmas()
                        # Checking that lemmas from relations are in the corpus if outside_corpus=False
                        syn_lemmas = [lemma for lemma in syn_lemmas if lemma in semcor_words]
                        # Adding each lemma to the list
                        for syn_lemma in syn_lemmas:
                            lemma_tuple = self.corpus_utilities.lemma_to_tuple(syn_lemma)
                            if word != lemma_tuple:
                                lemma_relations[relation].append(lemma_tuple)
                word_sem_rel_subdict = self.create_word_sem_rel_dict(synonyms=synonyms,
                                                                     hypernyms=lemma_relations[0],
                                                                     hyponyms=lemma_relations[1],
                                                                     holonyms=lemma_relations[2],
                                                                     meronyms=lemma_relations[3],
                                                                     attributes=lemma_relations[4],
                                                                     entailments=lemma_relations[5],
                                                                     causes=lemma_relations[6],
                                                                     also_sees=lemma_relations[7],
                                                                     verb_groups=lemma_relations[8],
                                                                     similar_tos=lemma_relations[9])
                # Adding pairs of word & the dictionary containing its relations to the big json list (since json doesn't let lists be keys)
                # But we can still keep the word_sem_rel_subdict intact since its keys are strings
                semantic_relations_list.append([word, word_sem_rel_subdict])
            sem_rel_file = open(sem_rel_path, 'w')
            json.dump(semantic_relations_list, sem_rel_file)
            sem_rel_file.close()
        semantic_relations_list = json.load(open(sem_rel_path))
        semantic_relations_dict = {}
        for pair in semantic_relations_list:
            key = tuple(pair[0])
            val_dict = pair[1]
            for val_key in ["synonyms", "hypernyms", "hyponyms", "holonyms", "meronyms", "attributes",
                            "entailments",
                            "causes", "also_sees", "verb_groups", "similar_tos"]:
                list_val_vals = val_dict[val_key]
                tuple_val_vals = []
                for val_val in list_val_vals:
                    tuple_val_vals.append(tuple(val_val))
                val_dict[val_key] = tuple_val_vals
            semantic_relations_dict[key] = val_dict
        return semantic_relations_dict


    def get_semantic_relations_word(self, word):
        """ Just getting the semantic relations dictionary for a single word in the corpus... but requiring that its
            relations be within the corpus sentence list. """
        semcor_words = set(sum(self.sentence_list, []))
        syn = wn_corpus.synset(word[1])
        synonyms = [self.corpus_utilities.lemma_to_tuple(synon) for synon in syn.lemmas() if
                    self.corpus_utilities.lemma_to_tuple(synon) != word]
        # These are all synsets.
        synset_relations = [syn.hypernyms(), syn.hyponyms(),
                            syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                            syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                            syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                            syn.verb_groups(), syn.similar_tos()]
        lemma_relations = []
        for relation in range(len(synset_relations)):
            lemma_relations.append([])
            # Getting each of the synsets above in synset_relations.
            for syn in range(len(synset_relations[relation])):
                # Getting the lemmas in each of the synset_relations synsets.
                syn_lemmas = synset_relations[relation][syn].lemmas()
                # Checking that lemmas from relations are in the corpus if outside_corpus=False
                syn_lemmas = [lemma for lemma in syn_lemmas if lemma in semcor_words]
                # Adding each lemma to the list
                for syn_lemma in syn_lemmas:
                    lemma_tuple = self.corpus_utilities.lemma_to_tuple(syn_lemma)
                    if word != lemma_tuple:
                        lemma_relations[relation].append(lemma_tuple)
        word_sem_rel_subdict = self.create_word_sem_rel_dict(synonyms=synonyms,
                                                             hypernyms=lemma_relations[0],
                                                             hyponyms=lemma_relations[1],
                                                             holonyms=lemma_relations[2],
                                                             meronyms=lemma_relations[3],
                                                             attributes=lemma_relations[4],
                                                             entailments=lemma_relations[5],
                                                             causes=lemma_relations[6],
                                                             also_sees=lemma_relations[7],
                                                             verb_groups=lemma_relations[8],
                                                             similar_tos=lemma_relations[9])
        return word_sem_rel_subdict


    def get_cooc_relations_dict(self):
        """
        Creates a dictionary of cooccurrence relations for every word in the corpus.
        Parameters:
              sentence_list (nested list): A list of sentences from the Semcor corpus.
              context_type (string): Indicates for cooccurrence dependent corpus mechanisms, whether we know the sense
                of the context words ("sense") or not ("word")
        Returns:
            (dict) Dictionary where each word in the corpus is a key and each of the other words it cooccurs with
                (are in the same sentence as our target word) are values (in a set).
        """
        cooc_rel_dict = defaultdict(set)
        for sent in self.sentence_list:
            for index in range(len(sent)):
                for context_index in range(len(sent)):
                    if index != context_index:
                        target_sense = sent[index]
                        context_sense = sent[context_index]
                        if self.context_type == "sense":
                            cooc_rel_dict[target_sense].update([context_sense])
                        else:
                            context_word = context_sense[0]
                            cooc_rel_dict[target_sense].update([context_word])
        return cooc_rel_dict


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


    def create_trial_network(self, words):
        """
                Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
                    hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
                     Note that all words are stored at time 1.
                Parameters:
                    words (list): list containing the target and context words for a given trial.
                Returns:
                    network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
                """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
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
        distance = 0 # Counter for figuring out how far away words are from the context/target words
        while curr_words:
            distance += 1
            for word in curr_words:
                # Checking for co-occurrent connections to add to network
                if distance <= self.cooc_depth and word in self.cooc_dict:
                    cooc_words = list(self.cooc_dict[word])
                else:
                    cooc_words = []
                # Checking for semantic relations to add (only in network for now)
                if word in self.semantic_relations_dict:
                    val_dict = self.semantic_relations_dict[word]
                    self.network.store(mem_id=word,
                                  time=self.time,
                                  spread_depth=spread_depth,
                                  synonyms=val_dict['synonyms'],
                                  hypernyms=val_dict['hypernyms'],
                                  hyponyms=val_dict['hyponyms'],
                                  holynyms=val_dict['holonyms'],
                                  meronyms=val_dict['meronyms'],
                                  attributes=val_dict['attributes'],
                                  entailments=val_dict['entailments'],
                                  causes=val_dict['causes'],
                                  also_sees=val_dict['also_sees'],
                                  verb_groups=val_dict['verb_groups'],
                                  similar_tos=val_dict['similar_tos'],
                                  cooc_words=cooc_words)
                    next_sem_relations = [item for sublist in list(val_dict.values()) for item in sublist]
                    next_words.update(cooc_words)
                    next_words.update(next_sem_relations)
                else:
                    self.network.store(mem_id=word,
                                      time=self.time,
                                      spread_depth=spread_depth,
                                      cooc_words=cooc_words)
                    next_words.update(cooc_words)
            prev_words.update(curr_words)
            # Making sure we don't redundantly add any words
            curr_words = next_words - prev_words
            next_words = set()


    def extend_trial_network(self, words):
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        curr_words = words
        prev_words = set()
        next_words = set()
        distance = 0  # Counter for figuring out how far away words are from the context/target words
        while curr_words:
            distance += 1
            for word in curr_words:
                # Checking if word is already in network. If so, go to next word.
                if word in self.network.knowledge:
                    continue
                # Checking for co-occurrent connections to add to network
                if distance <= self.cooc_depth and word in self.cooc_dict:
                    cooc_words = list(self.cooc_dict[word])
                else:
                    cooc_words = []
                # Checking for semantic relations to add (only in network for now)
                if word in self.semantic_relations_dict:
                    val_dict = self.semantic_relations_dict[word]
                    self.network.store(mem_id=word,
                                  time=self.time,
                                  spread_depth=spread_depth,
                                  synonyms=val_dict['synonyms'],
                                  hypernyms=val_dict['hypernyms'],
                                  hyponyms=val_dict['hyponyms'],
                                  holynyms=val_dict['holonyms'],
                                  meronyms=val_dict['meronyms'],
                                  attributes=val_dict['attributes'],
                                  entailments=val_dict['entailments'],
                                  causes=val_dict['causes'],
                                  also_sees=val_dict['also_sees'],
                                  verb_groups=val_dict['verb_groups'],
                                  similar_tos=val_dict['similar_tos'],
                                  cooc_words=cooc_words)
                    next_sem_relations = [item for sublist in list(val_dict.values()) for item in sublist]
                    next_words.update(cooc_words)
                    next_words.update(next_sem_relations)
                else:
                    self.network.store(mem_id=word,
                                  time=self.time,
                                  spread_depth=spread_depth,
                                  cooc_words=cooc_words)
                    next_words.update(cooc_words)
            prev_words.update(curr_words)
            # Making sure we don't redundantly add any words
            curr_words = next_words - prev_words
            next_words = set()


    def create_sem_network(self):
        """
                Builds a semantic network with each word in the Semcor corpus and its corresponding synonyms, hypernyms,
                    hyponyms, holonyms, meronyms, attributes, entailments, causes, also_sees, verb_groups, and similar_tos.
                     Note that all words are stored at time 1.
                Returns:
                    network (sentenceLTM): Semantic network with all words and co-occurrence relations in the Semcor corpus.
                """
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        semantic_relations_dict = self.get_semantic_relations_dict()
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
        relations_keys = list(semantic_relations_dict.keys())

        for word_index in range(len(relations_keys)):
            #if word_index % 10 == 0:
                #print(word_index, "out of", len(relations_keys))
            word_key = relations_keys[word_index]
            val_dict = semantic_relations_dict[word_key]
            cooc_words = list(self.cooc_dict[word_key])
            network.store(mem_id=word_key,
                          time=self.time,
                          spread_depth=spread_depth,
                          synonyms=val_dict['synonyms'],
                          hypernyms=val_dict['hypernyms'],
                          hyponyms=val_dict['hyponyms'],
                          holynyms=val_dict['holonyms'],
                          meronyms=val_dict['meronyms'],
                          attributes=val_dict['attributes'],
                          entailments=val_dict['entailments'],
                          causes=val_dict['causes'],
                          also_sees=val_dict['also_sees'],
                          verb_groups=val_dict['verb_groups'],
                          similar_tos=val_dict['similar_tos'],
                          cooc_words=cooc_words)
        return network


    def get_distribution(self, word_index, sentence):
        self.time += 1
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        word = sentence[word_index]
        max_act = float('-inf')
        max_guess = []

        target_senses = self.word_sense_dict[word[0]]
        words = set()
        words.update(target_senses)
        words.update(sentence)
        self.extend_trial_network(list(words))

        dist = {}

        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.network.store(mem_id=sentence[index], time=self.time, spread_depth=spread_depth)
            self.time += 1
        other_senses = self.word_sense_dict[word[0]]
        for candidate in other_senses:
            candidate_act = self.network.get_activation(mem_id=candidate, time=self.time)
            dist[candidate] = candidate_act
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        if self.activate_answer:
            self.network.store(mem_id=word, time=self.time, spread_depth=spread_depth)
        for elem in max_guess:
            self.network.store(mem_id=elem, time=self.time, spread_depth=spread_depth)
        self.time += 1
        return dist

    def guess(self, word_index, sentence):
        self.time += 1
        spread_depth = -1
        if not self.spreading:
            spread_depth = 0
        max_act = float('-inf')
        max_guess = []
        word = sentence[word_index]

        target_senses = self.word_sense_dict[word[0]]
        words = set()
        words.update(target_senses)
        words.update(sentence)
        self.extend_trial_network(list(words))

        if self.activate_sentence_words:  # Activating every other word in the sentence...
            for index in range(len(sentence)):
                if index != word_index:
                    self.network.store(mem_id=sentence[index], time=self.time, spread_depth=spread_depth)
            self.time += 1
        other_senses = self.word_sense_dict[word[0]]
        for candidate in other_senses:
            candidate_act = self.network.get_activation(mem_id=candidate, time=self.time)
            if candidate_act > max_act:
                max_act = candidate_act
                max_guess = [candidate]
            elif candidate_act == max_act:
                max_guess.append(candidate)
        if self.activate_answer:
            self.network.store(mem_id=word, time=self.time, spread_depth=spread_depth)
        for elem in max_guess:
            self.network.store(mem_id=elem, time=self.time, spread_depth=spread_depth)
        self.time += 1
        return max_guess


class AgentSpreadingBoostedCooc(AgentCooccurrence):

    def __init__(self, partition, num_sentences, context_type, func="sqrt"):
        super().__init__(partition, num_sentences, context_type)
        self.semantic_relations_dict = self.get_semantic_relations_dict()
        self.func = func

    def to_string_id(self):
        result = 'AgentSpreadingBoostedCooc'
        result += '_context_' + self.context_type
        result += '_func_' + self.func
        return result

    def get_semantic_relations_dict(self):
        """
            Gets the words related to each word in sentence_list and builds a dictionary to make the semantic network
            Parameters:
                sentence_list (list): list of all sentences or a partition of n sentences in the corpus
                partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                    at sentences 10000 - 14999.
                outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
                    relations are only considered from words inside the corpus.
            Returns:
                (dict) A dictionary with the semantic relations for every unique word in sentence_list
        """
        sem_rel_path = "./semantic_relations_lists/semantic_relations_list_inside_corpus"
        if len(self.sentence_list) == 30195:
            sem_rel_path = sem_rel_path + ".json"
        elif self.partition == 1:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + ".json"
        else:
            sem_rel_path = sem_rel_path + "_" + str(len(self.sentence_list)) + "_partition_" + str(
                self.partition) + ".json"
        if not os.path.isfile(sem_rel_path):
            semantic_relations_list = []
            # These are all the words in the corpus.
            semcor_words = set(sum(self.sentence_list, []))
            counter = 0
            for word in semcor_words:
                counter += 1
                syn = wn_corpus.synset(word[1])
                synonyms = [self.corpus_utilities.lemma_to_tuple(synon) for synon in syn.lemmas() if
                            self.corpus_utilities.lemma_to_tuple(synon) != word]
                # These are all synsets.
                synset_relations = [syn.hypernyms(), syn.hyponyms(),
                                    syn.member_holonyms() + syn.substance_holonyms() + syn.part_holonyms(),
                                    syn.member_meronyms() + syn.substance_meronyms() + syn.part_meronyms(),
                                    syn.attributes(), syn.entailments(), syn.causes(), syn.also_sees(),
                                    syn.verb_groups(), syn.similar_tos()]
                lemma_relations = []
                for relation in range(len(synset_relations)):
                    lemma_relations.append([])
                    # Getting each of the synsets above in synset_relations.
                    for syn in range(len(synset_relations[relation])):
                        # Getting the lemmas in each of the synset_relations synsets.
                        syn_lemmas = synset_relations[relation][syn].lemmas()
                        # Checking that lemmas from relations are in the corpus if outside_corpus=False
                        syn_lemmas = [lemma for lemma in syn_lemmas if lemma in semcor_words]
                        # Adding each lemma to the list
                        for syn_lemma in syn_lemmas:
                            lemma_tuple = self.corpus_utilities.lemma_to_tuple(syn_lemma)
                            if word != lemma_tuple:
                                lemma_relations[relation].append(lemma_tuple)
                word_sem_rel_subdict = self.create_word_sem_rel_dict(synonyms=synonyms,
                                                                     hypernyms=lemma_relations[0],
                                                                     hyponyms=lemma_relations[1],
                                                                     holonyms=lemma_relations[2],
                                                                     meronyms=lemma_relations[3],
                                                                     attributes=lemma_relations[4],
                                                                     entailments=lemma_relations[5],
                                                                     causes=lemma_relations[6],
                                                                     also_sees=lemma_relations[7],
                                                                     verb_groups=lemma_relations[8],
                                                                     similar_tos=lemma_relations[9])
                # Adding pairs of word & the dictionary containing its relations to the big json list (since json doesn't let lists be keys)
                # But we can still keep the word_sem_rel_subdict intact since its keys are strings
                semantic_relations_list.append([word, word_sem_rel_subdict])
            sem_rel_file = open(sem_rel_path, 'w')
            json.dump(semantic_relations_list, sem_rel_file)
            sem_rel_file.close()
        semantic_relations_list = json.load(open(sem_rel_path))
        semantic_relations_dict = {}
        for pair in semantic_relations_list:
            key = tuple(pair[0])
            val_dict = pair[1]
            for val_key in ["synonyms", "hypernyms", "hyponyms", "holonyms", "meronyms", "attributes",
                            "entailments",
                            "causes", "also_sees", "verb_groups", "similar_tos"]:
                list_val_vals = val_dict[val_key]
                tuple_val_vals = []
                for val_val in list_val_vals:
                    tuple_val_vals.append(tuple(val_val))
                val_dict[val_key] = tuple_val_vals
            semantic_relations_dict[key] = val_dict
        return semantic_relations_dict

    def create_word_sem_rel_dict(self, synonyms, hypernyms, hyponyms, holonyms, meronyms, attributes,
                                 entailments, causes, also_sees, verb_groups, similar_tos):
        """
        Creates a semantic relations dictionary with given semantic relations for a word.
        Parameters:
            synonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hypernyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            hyponyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            holonyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            meronyms (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            attributes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            entailments (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            causes (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            also_sees (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            verb_groups (list) A list of word relations drawn from the synset a word belongs to from the nltk package
            similar_tos (list) A list of word relations drawn from the synset a word belongs to from the nltk package
        Returns: A dictionary with the semantic relations for one word in the corpus.
        """
        sem_rel_dict = {"synonyms": set(synonyms), "hypernyms": set(hypernyms), "hyponyms": set(hyponyms),
                        "holonyms": set(holonyms), "meronyms": set(meronyms), "attributes": set(attributes),
                        "entailments": set(entailments), "causes": set(causes), "also_sees": set(also_sees),
                        "verb_groups": set(verb_groups), "similar_tos": set(similar_tos)}
        for rel in sem_rel_dict.keys():
            vals = sem_rel_dict[rel]
            string_vals = []
            for val in vals:
                string_vals.append(list(val))
            sem_rel_dict[rel] = string_vals
        return sem_rel_dict

    def change_probability(self, probability):
        if self.func == 'sqrt':
            return math.sqrt(probability)
        elif self.func == 'sigmoid':
            return 1 / (1 + math.exp(-probability))

    def find_probability(self, target, base):
        cooc_prob = self.get_conditional_probability(target=target, base=base)
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

    def get_distribution(self, word_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
        dist = {}
        target_sense = sentence[word_index]
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            for context_index in range(len(sentence)):
                if context_index != word_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.find_probability(target=target_sense_candidate,
                                                                                          base=context_word))
            dist[target_sense_candidate] = candidate_conditional_probability
        return dist

    def guess(self, word_index, sentence):
        """
        Completes the WSD task.
        Parameters:
            target_index (int): Integer >= 0 corresponding to the index of the list of sentence words where the target
                sense can be found.
            sentence (list): List of words in the current sentence from the SemCor corpus.
        Returns:
            (list) A list of word sense disambiguation sense guesses.
        """
        max_score = -float("inf")
        max_senses = None
        target_sense = sentence[word_index]
        if self.context_type == "word":
            sentence = [word[0] for word in sentence]
        for target_sense_candidate in self.word_sense_dict[target_sense[0]]:
            candidate_conditional_probability = 1
            for context_index in range(len(sentence)):
                if context_index != word_index:
                    context_word = sentence[context_index]
                    candidate_conditional_probability = (candidate_conditional_probability *
                                                         self.find_probability(target=target_sense_candidate,
                                                                                          base=context_word))
            if candidate_conditional_probability > max_score:
                max_score = candidate_conditional_probability
                max_senses = [target_sense_candidate]
            elif candidate_conditional_probability == max_score:
                max_senses.append(target_sense_candidate)
        return max_senses


class AgentOracle(WSDAgentTemplate):
    def __init__(self, partition, num_sentences, activation_base=2, decay_parameter=0.05, constant_offset=0,
                 activate_answer=False, activate_sentence_words=False):
        """
        Parameters:
            corpus_utilities (class): A class of functions useful for corpus mechanisms, specific to the partition of the
                Semcor corpus used
            outside_corpus (bool): True if semantic relations can be considered outside the corpus and False if semantic
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        self.partition = partition
        self.num_sentences = num_sentences
        self.corpus_utilities = CorpusUtilities(num_sentences, partition)
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter
        self.constant_offset = constant_offset
        self.activate_answer = activate_answer
        self.activate_sentence_words = activate_sentence_words
        self.sentence_list = self.corpus_utilities.get_sentence_list()
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()
        self.sem_nospread_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=False,
                                                 clear="never", activation_base=activation_base,
                                                 decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                 activate_answer=activate_answer,
                                                 activate_sentence_words=activate_sentence_words, bounded=False)
        self.sem_never_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=True,
                                              clear="never",
                                              activation_base=activation_base, decay_parameter=decay_parameter,
                                              constant_offset=constant_offset, activate_answer=activate_answer,
                                                 activate_sentence_words=activate_sentence_words, bounded=False)
        self.sem_sentence_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=True,
                                                 clear="sentence", activation_base=activation_base,
                                                 decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                 activate_answer=activate_answer,
                                                 activate_sentence_words=activate_sentence_words, bounded=False)
        self.sem_word_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=True,
                                             clear="word",
                                             activation_base=activation_base, decay_parameter=decay_parameter,
                                             constant_offset=constant_offset,
                                             activate_answer=activate_answer,
                                             activate_sentence_words=activate_sentence_words, bounded=False)
        self.sem_never_bounded_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=True,
                                              clear="never",
                                              activation_base=activation_base, decay_parameter=decay_parameter,
                                              constant_offset=constant_offset, activate_answer=activate_answer,
                                              activate_sentence_words=activate_sentence_words, bounded=True)
        self.sem_sentence_bounded_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=True,
                                                 clear="sentence", activation_base=activation_base,
                                                 decay_parameter=decay_parameter, constant_offset=constant_offset,
                                                 activate_answer=activate_answer,
                                                 activate_sentence_words=activate_sentence_words, bounded=True)
        self.sem_word_bounded_agent = AgentSpreading(num_sentences=num_sentences, partition=partition, spreading=True,
                                             clear="word",
                                             activation_base=activation_base, decay_parameter=decay_parameter,
                                             constant_offset=constant_offset,
                                             activate_answer=activate_answer,
                                             activate_sentence_words=activate_sentence_words, bounded=True)
        self.cooc_word_agent = AgentCooccurrence(num_sentences=num_sentences,
                                                 partition=partition,
                                                 context_type="word")
        self.cooc_sense_agent = AgentCooccurrence(num_sentences=num_sentences,
                                                  partition=partition,
                                                  context_type="sense")

    def to_string_id(self):
        result = "AgentOracle"
        return result

    def guess(self, word_index, sentence):
        """
        Completes a trial of the WSD.
        Parameters:
            target_index (int): The index of the "target" word in the sentence given in the sentence parameter list.
            sentence (list): A list of lemma/synset tuples referring to all words in the sentence (including the target
                sense)
            timer_word (int): Timer for the network that clears after every word.
            timer_sentence (int): Timer for the network that clears after every sentence.
            timer_never (int): Timer for the network that never clears.
        """
        correct_sense = sentence[word_index]
        for agent in [self.cooc_word_agent, self.cooc_sense_agent, self.sem_nospread_agent, self.sem_word_agent,
                      self.sem_word_bounded_agent, self.sem_sentence_agent, self.sem_sentence_bounded_agent,
                      self.sem_never_agent, self.sem_never_bounded_agent]:
            guess = agent.guess(word_index, sentence)
            if correct_sense in guess:
                return [correct_sense]
        return []
        # cooc_word_guess = self.cooc_word_agent.guess(word_index, sentence)
        # if correct_sense in cooc_word_guess:
        #     return [correct_sense]
        # cooc_sense_guess = self.cooc_sense_agent.guess(word_index, sentence)
        # if correct_sense in cooc_sense_guess:
        #     return [correct_sense]
        # sem_guess_no_spread = self.sem_nospread_agent.guess(word_index, sentence)
        # if correct_sense in sem_guess_no_spread:
        #     return [correct_sense]
        # sem_guess_never = self.sem_never_agent.guess(word_index, sentence)
        # if correct_sense in sem_guess_never:
        #     return [correct_sense]
        # sem_guess_sentence = self.sem_sentence_agent.guess(word_index, sentence)
        # if correct_sense in sem_guess_sentence:
        #     return [correct_sense]
        # sem_guess_word = self.sem_word_agent.guess(word_index, sentence)
        # if correct_sense in sem_guess_word:
        #     return [correct_sense]
        # else:
        #     return [None]


class CorpusUtilities:
    """ A small library of functions that assist in working with the Semcor corpus"""

    def __init__(self, num_sentences=-1, partition=1):
        """
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking
                at sentences 10000 - 14999.
        """
        self.num_sentences = num_sentences
        self.partition = partition

    def lemma_to_tuple(self, lemma):
        """
        Converts lemmas to tuples to prevent usage of the nltk corpus
        Parameters:
            lemma (lemma object) a lemma object from the nltk package
        Returns:
            (tuple) a tuple containing the sense and synset of the word originally in lemma format.
        """
        lemma_word = lemma.name()
        synset_string = lemma.synset().name()
        lemma_tuple = (lemma_word, synset_string)
        return lemma_tuple

    def get_sentence_list(self):
        """
        Gets sentence list from semcor corpus in nltk python module
        Parameters:
            num_sentences (int): The number of sentences from the corpus to use in the task. The first n sentences
                from the corpus are used and if n=-1, all sentences from the corpus are used.
            partition (int): The subset of sentences to consider. i.e. if n=5000, and partition = 2, we would be looking at
                sentences 10000 - 14999.
        Returns:
            (list) sentence_list (list of all sentences or the first n sentences of the corpus)
        """
        if self.num_sentences == -1:
            sentence_list_path = "./sentence_list/sentence_list.json"
        elif self.num_sentences != -1 and self.partition == 1:
            sentence_list_path = "./sentence_list/sentence_list_" + str(self.num_sentences) + ".json"
        else:
            sentence_list_path = "./sentence_list/sentence_list_" + str(self.num_sentences) + "_partition_" + str(
                self.partition) + ".json"
        if not os.path.isfile(sentence_list_path):
            # Checking that file exists
            sentence_list = []
            if self.num_sentences == -1:
                semcor_sents = semcor.tagged_sents(tag="sem")
            else:
                if self.partition == 1:
                    semcor_sents = semcor.tagged_sents(tag="sem")[0:self.num_sentences]
                elif self.partition * self.num_sentences > 30195:
                    raise ValueError(self.partition, self.num_sentences)
                else:
                    semcor_sents = semcor.tagged_sents(tag="sem")[
                                   (self.num_sentences * (self.partition - 1)):(self.num_sentences * self.partition)]
            for sentence in semcor_sents:
                sentence_word_list = []
                for item in sentence:
                    if not isinstance(item, nltk.Tree):
                        continue
                    if not isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                        continue
                    corpus_word = self.lemma_to_tuple(item.label())
                    sentence_word_list.append(corpus_word)
                if len(sentence_word_list) > 1:
                    sentence_list.append(sentence_word_list)
            sent_list_file = open(sentence_list_path, 'w')
            json.dump(sentence_list, sent_list_file)
            sent_list_file.close()
        else:
            # Getting json file containing the sentence list and converting the words stored as strings into tuples
            sentence_list = json.load(open(sentence_list_path))
            for sentence_index in range(len(sentence_list)):
                for word_index in range(len(sentence_list[sentence_index])):
                    word = sentence_list[sentence_index][word_index]
                    sentence_list[sentence_index][word_index] = tuple(word)
        return sentence_list

    def get_corpus_sentence_list(self):
        sentence_list_path = "./sentence_list/sentence_list.json"
        if not os.path.isfile(sentence_list_path):
            # Checking that file exists
            sentence_list = []
            if self.num_sentences == -1:
                semcor_sents = semcor.tagged_sents(tag="sem")
            else:
                if self.partition == 1:
                    semcor_sents = semcor.tagged_sents(tag="sem")[0:self.num_sentences]
                elif self.partition * self.num_sentences > 30195:
                    raise ValueError(self.partition, self.num_sentences)
                else:
                    semcor_sents = semcor.tagged_sents(tag="sem")[
                                   (self.num_sentences * (self.partition - 1)):(self.num_sentences * self.partition)]
            for sentence in semcor_sents:
                sentence_word_list = []
                for item in sentence:
                    if not isinstance(item, nltk.Tree):
                        continue
                    if not isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                        continue
                    corpus_word = self.lemma_to_tuple(item.label())
                    sentence_word_list.append(corpus_word)
                if len(sentence_word_list) > 1:
                    sentence_list.append(sentence_word_list)
            sent_list_file = open(sentence_list_path, 'w')
            json.dump(sentence_list, sent_list_file)
            sent_list_file.close()
        else:
            # Getting json file containing the sentence list and converting the words stored as strings into tuples
            sentence_list = json.load(open(sentence_list_path))
            for sentence_index in range(len(sentence_list)):
                for word_index in range(len(sentence_list[sentence_index])):
                    word = sentence_list[sentence_index][word_index]
                    sentence_list[sentence_index][word_index] = tuple(word)
        return sentence_list

    def get_word_counts(self):
        """ Gets the number of times each word (encompassing all senses) occurs in the sentence list"""
        word_counts = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for sense in sentence:
                word_counts[sense[0]] += 1
        return word_counts

    def get_sense_counts(self):
        """ Gets the number of times each sense-specific word occurs in the sentence list"""
        sense_counts = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for sense in sentence:
                sense_counts[sense] += 1
        return sense_counts

    def get_word_word_cooccurrences(self):
        """ Creates a symmetric dictionary keys as word/word tuples and values the number of times each occur in the
         same sentence"""
        word_word_cooccurrences = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                target_sense = sentence[target_index]
                target_word = target_sense[0]
                for other_index in range(len(sentence)):
                    if target_index != other_index:
                        other_sense = sentence[other_index]
                        other_word = other_sense[0]
                        word_word_cooccurrences[(target_word, other_word)] += 1
        return word_word_cooccurrences

    def get_sense_sense_cooccurrences(self):
        """ Creates a symmetric dictionary keys as sense/sense tuples and values the number of times each occur in the
                 same sentence"""
        sense_sense_cooccurrences = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                target_sense = sentence[target_index]
                for other_index in range(len(sentence)):
                    if target_index != other_index:
                        other_sense = sentence[other_index]
                        sense_sense_cooccurrences[(target_sense, other_sense)] += 1
        return sense_sense_cooccurrences

    def get_sense_word_cooccurrences(self):
        """ Creates a symmetric dictionary keys as sense/word tuples and values the number of times each occur in the
                 same sentence"""
        sense_word_cooccurrences = defaultdict(int)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            for target_index in range(len(sentence)):
                target_sense = sentence[target_index]
                for other_index in range(len(sentence)):
                    if target_index != other_index:
                        other_sense = sentence[other_index]
                        other_word = other_sense[0]
                        sense_word_cooccurrences[(target_sense, other_word)] += 1
        return sense_word_cooccurrences

    def get_word_sense_dict(self):
        """
        Makes a dictionary with each senseless word the key, and each of its senses the values.
        Returns:
             (dict) dictionary with the possible senses of each word in the corpus
            """
        word_sense_dict = defaultdict(set)
        sentence_list = self.get_sentence_list()
        for sentence in sentence_list:
            temp_word_sense_dict = defaultdict(set)
            for word in sentence:
                temp_word_sense_dict[word[0]].add(word)
            if len(temp_word_sense_dict) > 1:
                for word, senses in temp_word_sense_dict.items():
                    word_sense_dict[word] = set(word_sense_dict[word])
                    word_sense_dict[word] |= senses
                    word_sense_dict[word] = list(word_sense_dict[word])
        return word_sense_dict


class WSDTask:
    def __init__(self, num_sentences=30195, partition=1):
        self.partition = partition
        self.num_sentences = num_sentences
        self.corpus_utilities = CorpusUtilities(partition=self.partition, num_sentences=self.num_sentences)
        self.word_sense_dict = self.corpus_utilities.get_word_sense_dict()

    def to_string_id(self):
        result = 'WSD'
        result += '_partition_' + str(self.partition)
        result += '_sents_' + str(self.num_sentences)
        return result

    def uniform_random(self):
        sentence_list = self.corpus_utilities.get_sentence_list()
        # collect results
        accuracy_list = []
        counter = 0
        for sentence in sentence_list:
            counter += 1
            #print("sentence", counter)
            for target in sentence:
                senses = wn_corpus.synsets(target[0])
                accuracy_list.append(len(senses))
        # save results to file
        return sum(accuracy_list)/len(accuracy_list)

    def is_correct(self, target, guesses):
        """Determines whether guesses to the sense of the target word are correct (True) or incorrect (False)"""
        correct_list = []
        for guess in guesses:
            if guess == target:
                correct_list.append(True)
            else:
                correct_list.append(False)
        return correct_list

    def save_results(self, agent, accuracy_list):
        print(accuracy_list)
        filename = self.to_string_id() + '_' + agent.to_string_id()
        with open('results/' + filename + ".json", "w") as fd:
            json.dump(accuracy_list, fd)
            fd.close()

    def save_distribution(self, agent, avg_dist):
        avg_dist_list = []
        #print(avg_dist)
        for trial in avg_dist:
            sublist = []
            target_word = trial[0]
            trial_candidates = trial[1]
            for item in trial_candidates:
                sublist.append([[item[0][0], item[0][1]], item[1]])
            avg_dist_list.append([[target_word[0], target_word[1]],sublist])
        filename = self.to_string_id() + '_' + agent.to_string_id()
        with open('agent_distributions/list_' + filename + ".json", "w") as fd:
            json.dump(avg_dist_list, fd)
            fd.close()

    def run(self, agent):
        #print("running...")
        sentence_list = self.corpus_utilities.get_sentence_list()
        # initialize the agent (e.g. pre-activate concepts)
        agent.setup_run()
        # collect results
        accuracy_list = []
        counter = 0
        for sentence in sentence_list:
            counter += 1
            #if counter % 50 == 0:
                #print("On sentence...", counter)
            agent.new_sentence_callback()
            for target_index in range(len(sentence)):
                agent.new_word_callback()
                guesses = agent.guess(word_index=target_index, sentence=sentence)
                correct = self.is_correct(sentence[target_index], guesses)
                accuracy_list.append([guesses, correct])
        # save results to file
        self.save_results(agent, accuracy_list)
        return accuracy_list

    def run_stats(self, percent_list, num_context_list):
        plt.scatter(num_context_list, percent_list)
        plt.show()

    def get_wsd_agent_distribution(self, agent):
        # print("running...")
        sentence_list = self.corpus_utilities.get_sentence_list()
        # initialize the agent (e.g. pre-activate concepts)
        agent.setup_run()
        # collect distributions in nested dictionary
        dist = []
        counter = 0
        for sentence in sentence_list:
            counter += 1
            #if counter % 50 == 0:
                #print("On sentence...", counter)
            agent.new_sentence_callback()
            for target_index in range(len(sentence)):
                agent.new_word_callback()
                trial_dist_dict = agent.get_distribution(word_index=target_index, sentence=sentence)
                trial_dist_list = []
                for elem in trial_dist_dict:
                    trial_dist_list.append([elem, trial_dist_dict[elem]])
                dist.append([sentence[target_index], trial_dist_list])
        # save results to file
        self.save_distribution(agent, dist)
        return dist


def create_WSD_agent(guess_method, partition=1, num_sentences=5000, context_type='sense', spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0, whole_corpus=False, threshold=0,
                 var_type="stdev", discount=0.1, activate_answer=False, activate_sentence_words=True, bounded=False,
                     num_context_acts=1, cooc_depth=1, func='sqrt'):
    if guess_method == 'spreading':
        return AgentSpreading(partition, num_sentences, spreading, clear, activation_base, decay_parameter,
                              constant_offset, activate_answer=activate_answer,
                              activate_sentence_words=activate_sentence_words, bounded=bounded,
                              num_context_acts=num_context_acts)
    elif guess_method == 'cooccurrence':
        return AgentCooccurrence(partition=partition, num_sentences=num_sentences, context_type=context_type)
    elif guess_method == 'cooc_thresh_sem':
        return AgentCoocThreshSpreading(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                        spreading=spreading, clear=clear, activation_base=activation_base,
                                        decay_parameter=decay_parameter, constant_offset=constant_offset,
                                        activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                        whole_corpus=whole_corpus, bounded=bounded, threshold=threshold)
    elif guess_method == 'sem_thresh_cooc':
        return AgentSpreadingThreshCooc(partition=partition, num_sentences=num_sentences, context_type=context_type)
    elif guess_method == 'joint_prob':
        return AgentJointProbability(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                     spreading=spreading, clear=clear, activation_base=activation_base,
                                     decay_parameter=decay_parameter, constant_offset=constant_offset,
                                     activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                     bounded=bounded)
    elif guess_method == 'add_prob':
        return AgentAdditiveProbability(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                     spreading=spreading, clear=clear, activation_base=activation_base,
                                     decay_parameter=decay_parameter, constant_offset=constant_offset,
                                     activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                     bounded=bounded)
    elif guess_method == 'joint_var':
        return AgentJointVariance(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                     clear=clear, activation_base=activation_base,
                                     decay_parameter=decay_parameter, constant_offset=constant_offset,
                                     activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                     bounded=bounded, var_type=var_type)
    elif guess_method == 'max_prob':
        return AgentMaxProbability(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                     spreading=spreading, clear=clear, activation_base=activation_base,
                                     decay_parameter=decay_parameter, constant_offset=constant_offset,
                                     activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                     bounded=bounded)
    elif guess_method == 'oracle':
        return AgentOracle(partition=partition, num_sentences=num_sentences, activation_base=activation_base,
                           decay_parameter=decay_parameter, constant_offset=constant_offset, activate_answer=activate_answer,
                           activate_sentence_words=activate_sentence_words)
    elif guess_method == 'cooc_weight_spreading':
        return AgentCoocWeightedSpreading(partition=partition, num_sentences=num_sentences, spreading=spreading,
                                          clear=clear, context_type=context_type, activation_base=activation_base,
                                          decay_parameter=decay_parameter, constant_offset=constant_offset,
                                          activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                          bounded=bounded)
    elif guess_method == 'bounded_spreading':
        return AgentBoundedSpreading(partition, num_sentences, spreading, clear, activation_base, decay_parameter,
                              constant_offset)
    elif guess_method == 'spread_supplemented_cooc':
        return AgentSpreadingSupplementedCooc(partition=partition, num_sentences=num_sentences,
                                              context_type=context_type, discount=discount)
    elif guess_method == 'cooc_supplemented_spreading':
        return AgentCoocSupplementedSpreading(partition=partition, num_sentences=num_sentences, spreading=spreading,
                                              clear=clear, activation_base=activation_base,
                                              decay_parameter=decay_parameter, constant_offset=constant_offset,
                                              context_type=context_type, activate_answer=activate_answer,
                                              activate_sentence_words=activate_sentence_words, bounded=bounded,
                                              discount=discount)
    elif guess_method == 'cooc_expanded_spreading':
        return AgentCoocExpandedSpreading(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                          spreading=spreading, clear=clear, activation_base=activation_base,
                                          decay_parameter=decay_parameter, constant_offset=constant_offset,
                                          activate_answer=activate_answer, activate_sentence_words=activate_sentence_words,
                                          bounded=bounded, cooc_depth=cooc_depth)
    elif guess_method == 'spreading_boosted_cooc':
        return AgentSpreadingBoostedCooc(partition=partition, num_sentences=num_sentences, context_type=context_type,
                                         func=func)


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
    for guess_method in ['joint_prob']:
        for partition in [1, 2, 3, 4, 5, 6]:
            # create WSD task
            wsd = WSDTask(num_sentences=5000, partition=partition)
            # create agent
            agent = create_WSD_agent(guess_method, num_sentences=5000, partition=partition,
                                 activate_answer=False, activate_sentence_words=True)
            # run (and saving data to disk)
            results = wsd.run(agent)
            # analyze
            # print("results: ", results)
            #print("weighted sum: ", analyze_data(results))

# Testing different frequencies of network clearing for sem spreading
def main2(num_acts, clear_type):
    for num_context_acts in range(num_acts):
        cumulative_acc = 0
        num_words = 0
        for partition in [1,2,3,4,5,6]:
            #print("partition:", partition)
            # create WSD task
            wsd = WSDTask(num_sentences=5000, partition=partition)
            # create agent
            agent = create_WSD_agent("spreading", num_sentences=5000, partition=partition, clear=clear_type,
                                 activate_answer=False, activate_sentence_words=True, num_context_acts=num_context_acts)
            results = wsd.run(agent)
            total_guesses = 0
            discounted_count = 0
            for word_guess in results:
                guesses = word_guess[1]
                for guess in guesses:
                    total_guesses += 1
                if any(guesses):
                    discounted_count += 1 / len(guesses)
            partition_words = len(results)
            num_words += partition_words
            cumulative_acc += (discounted_count / total_guesses) * partition_words
        #print("Times context activated:", num_context_acts, "accuracy:", cumulative_acc / num_words)

# Testing different discounts for CSS
def main3(discounts, clear_type):
    for discount in discounts:
        cumulative_acc = 0
        num_words = 0
        for partition in [1, 2, 3, 4, 5, 6]:
            print("partition:", partition)
            # create WSD task
            wsd = WSDTask(num_sentences=5000, partition=partition)
            # create agent
            agent = create_WSD_agent("cooc_supplemented_spread", num_sentences=5000, partition=partition, clear=clear_type,
                                     activate_answer=False, activate_sentence_words=True, discount=discount)
            results = wsd.run(agent)
            total_guesses = 0
            discounted_count = 0
            for word_guess in results:
                guesses = word_guess[1]
                for guess in guesses:
                    total_guesses += 1
                if any(guesses):
                    discounted_count += 1 / len(guesses)
            partition_words = len(results)
            num_words += partition_words
            cumulative_acc += (discounted_count / total_guesses) * partition_words
        print("Times context activated:", discount, "clear type", clear_type, "accuracy:", cumulative_acc / num_words)

def main_dists():
    for guess_method in ['cooc_supplemented_spreading']:
        for partition in [1,2,3,4,5,6]:
            for bounded in [False, True]:
                for clear in ["never", "sentence", 'word']:
                    for context in ["word", "sense"]:
                        # create WSD task
                        print("partition", partition, "bounded", bounded, "clear", clear, "context", context)
                        wsd = WSDTask(num_sentences=5000, partition=partition)
                        # create agent
                        agent = create_WSD_agent(guess_method, num_sentences=5000, partition=partition, context_type=context,
                                 activate_answer=False, activate_sentence_words=True, clear=clear, bounded=bounded)
                        # run (and saving data to disk)
                        dist = wsd.get_wsd_agent_distribution(agent)
                        # analyze
main_dists()