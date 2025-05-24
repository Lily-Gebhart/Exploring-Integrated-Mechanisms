""" Gets the distribution for spreading and cooccurrence on RAT and WSD specific corpora/networks"""

from integrated_mechanisms_WSD import AgentJointProbability as WSD_AgentJointProbability
from integrated_mechanisms_WSD import create_WSD_agent, WSDTask
from integrated_mechanisms_RAT import create_RAT_agent, RatTest
from integrated_mechanisms_RAT import AgentJointProbability as RAT_AgentJointProbability
from n_gram_cooccurrence.google_ngrams import *
import csv, json, os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import math
import matplotlib.ticker as mtick
from pareto_plot import get_wsd_filenames


def get_baseline_distribution(task, type, context_type, clear="never"):
    """ Gets the non-averaged distribution for each trial of the specified task (WSD, RAT), specified mechanism
     (cooccurrence, spreading), and subtype (word/sense, never/word/sentence, NA, SFFAN/SWOWEN/combined)"""
    filename = "/Users/lilygebhart/Downloads/research/distributions/" + str(task) + "_" + str(type) + "_" + str(context_type) + "_" + str(clear) + "_non_averaged_distributions.json"
    if (not os.path.isfile(filename)):
        distributions = []
        if task == "WSD":  # Task is WSD
            agent = WSD_AgentJointProbability(context_type=context_type, clear=clear, num_sentences=-1, partition=1)
            sentence_list = agent.corpus_utilities.get_sentence_list()
            word_sense_dict = agent.corpus_utilities.get_word_sense_dict()
            if type == "cooccurrence":  # Type is cooccurrence
                len_sents = len(sentence_list)
                counter = 0
                for sent in sentence_list:
                    counter += 1
                    #if counter % 50 == 0:
                        #print(counter, "out of", len_sents)
                    for word_index in range(len(sent)):
                        word = sent[word_index]
                        word_senses = word_sense_dict[word[0]]
                        dist = agent.get_cooccurrence_distribution(word_index, word_senses, sent, context_type)
                        dist_total = sum(dist.values())
                        normalized_dist = []
                        for prob in dist.values():
                            normalized_dist.append(prob / dist_total)
                        distributions.append(sorted(normalized_dist))
            else:  # Type is semantic spreading
                timer = 2
                counter = 0
                len_sents = len(sentence_list)
                for sent in sentence_list:
                    counter += 1
                    #if counter % 50 == 0:
                        #print(counter, "out of", len_sents)
                    if clear == "sentence":
                        agent.new_sentence_callback()
                        timer = 2
                    for word_index in range(len(sent)):
                        word = sent[word_index]
                        word_senses = word_sense_dict[word[0]]
                        dist = agent.get_spreading_distribution(word_senses=word_senses, time=timer)
                        distributions.append(sorted(list(dist.values())))
                        if clear != "word":
                            max_spread = -float("inf")
                            guesses = []
                            for key in list(dist.keys()):
                                prob = dist[key]
                                if prob > max_spread:
                                    guesses = [key]
                                    max_spread = prob
                                if prob == max_spread:
                                    guesses.append(key)
                            for guess in guesses:
                                agent.spreading_agent.network.store(guess, timer)
                            agent.spreading_agent.network.store(word, timer)
                            timer += 1
        else:  # Task is RAT
            with open('./nltk_english_stopwords', "r") as stopwords_file:
                lines = stopwords_file.readlines()
                stopwords = []
                for l in lines:
                    stopwords.append(l[:-1])
            agent = RAT_AgentJointProbability(source="SFFAN", spreading=True)
            rat_file = csv.reader(open('./RAT/RAT_items.txt'))
            next(rat_file)
            trial_counter = 0
            for trial in rat_file:
                trial_counter += 1
                #print("trial", trial_counter)
                context = [trial[0].upper(), trial[1].upper(), trial[2].upper()]
                if type == "cooccurrence":  # Type is cooccurrence
                    dist = agent.get_cooccurrence_distribution(context[0], context[1], context[2])
                else:  # Type is semantic spreading
                    dist = agent.get_spreading_distribution(context[0], context[1], context[2])
                distributions.append(sorted(list(dist.values())))
        file = open(filename, 'w')
        json.dump(distributions, file)
        file.close()
    else:
        distributions = json.load(open(filename))
    return distributions

def get_average_baseline_distribution(task, type, context_type="", clear="", num_points=100):
    filename = "/Users/lilygebhart/Downloads/research/distributions/" + str(task) + "_" + str(type) + "_" + str(context_type) + "_" + str(clear) + "_averaged_distributions.json"
    if not os.path.isfile(filename):
        dists = get_baseline_distribution(task, type, context_type, clear)
        avg_dists = [0] * num_points
        #print(avg_dists)
        num_dists = len(dists)
        x_vals = np.linspace(0, 1, num_points)
        counter = 0
        for dist in dists:  # Getting each distribution
            counter += 1
            #print(counter)
            if len(dist) == 0:
                num_dists -= 1
                continue
            elif len(dist) == 1:
                dist = [dist[0], dist[0]]
                #print(dist)
            dist_x_vals = np.linspace(0, 1, len(dist))
            y_interp = interp1d(dist_x_vals, dist)
            for i in range(num_points):
                # if i == 0:
                #     continue
                # elif i == (num_points - 1):
                #     avg_dists[num_points - 1] += dist[-1]
                #else:
                avg_dists[i] += y_interp(x_vals[i])
        avg_dist = [list(x_vals), [elem/num_dists for elem in avg_dists]]
        #print("interpolation", avg_dist[1])
        file = open(filename, 'w')
        json.dump(avg_dist, file)
        file.close()
    else:
        avg_dist = json.load(open(filename))
    return avg_dist

def plot_baseline_distributions(task, num_points):
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("Normalized, Ranked Words")
    plt.ylabel("Average Conditional Probability")
    if task == "WSD":
        plt.title("Average Distributions for Spreading & Co-Occurrence on the WSD")
        cooc_word_dist = get_average_baseline_distribution(task, "cooccurrence", context_type="word", num_points=num_points)
        cooc_sense_dist = get_average_baseline_distribution(task, "cooccurrence", context_type="sense", num_points=num_points)
        sem_never_dist = get_average_baseline_distribution(task, "spreading", clear="never", num_points=num_points)
        sem_word_dist = get_average_baseline_distribution(task, "spreading", clear="word", num_points=num_points)
        sem_sent_dist = get_average_baseline_distribution(task, "spreading", clear="sentence", num_points=num_points)
        plt.plot(cooc_word_dist[0][1:-1], cooc_word_dist[1][1:-1], label="Co-occurrence: Context Word",
                color="tab:blue")
        plt.plot(cooc_sense_dist[0][1:-1], cooc_sense_dist[1][1:-1], label="Co-occurrence: Context Sense",
                color="tab:green")
        plt.plot(sem_never_dist[0][1:-1], sem_never_dist[1][1:-1], label="Spreading: Clear Never",
                color="tab:orange")
        plt.plot(sem_sent_dist[0][1:-1], sem_sent_dist[1][1:-1], label="Spreading: Clear Sentence",
                color="tab:red")
        plt.plot(sem_word_dist[0][1:-1], sem_word_dist[1][1:-1], label="Spreading: Clear Word",
                color="tab:pink")
    if task == "RAT":
        plt.title("Average Distributions for Spreading & Co-Occurrence on the RAT")
        cooc_dist = get_average_baseline_distribution(task, "cooccurrence", num_points=num_points)
        sem_SFFAN_dist = get_average_baseline_distribution(task, "spreading", num_points=num_points)
        plt.plot(cooc_dist[0], cooc_dist[1], label="Co-occurrence", color="tab:blue")
        plt.plot(sem_SFFAN_dist[0], sem_SFFAN_dist[1], label="Spreading: SFFAN", color="tab:orange")
    plt.legend(loc="upper left")
    plt.show()

def get_integrated_distribution(task, agent, context_type="word", clear="never", whole_corpus=False, threshold=0,
                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1, func='sqrt'):
    distributions = []
    if task == 'wsd':
        for partition in range(1, 7):
            agent_id = create_WSD_agent(agent, partition=partition, num_sentences=5000, context_type=context_type,
                                     spreading=True, clear=clear, activation_base=2, decay_parameter=0.05,
                                     constant_offset=0, whole_corpus=whole_corpus, threshold=threshold,
                                     var_type=var_type, discount=discount, activate_answer=False,
                                     activate_sentence_words=True, bounded=bounded, num_context_acts=num_context_acts,
                                     cooc_depth=cooc_depth, func=func)
            # get the agent distribution...
            wsd = WSDTask(num_sentences=5000, partition=partition)
            filename = "agent_distributions/list_" + wsd.to_string_id() + "_" + agent_id.to_string_id() + ".json"
            if not os.path.isfile(filename):
                dist = wsd.get_wsd_agent_distribution(agent_id)
                dist_list = []
                for trial in dist:
                    sublist = []
                    target_word = trial[0]
                    trial_candidates = trial[1]
                    for item in trial_candidates:
                        sublist.append([[item[0][0], item[0][1]], item[1]])
                    dist_list.append([[target_word[0], target_word[1]], sublist])
                distributions.append(dist_list)
                file = open(filename, 'w')
                json.dump(dist_list, file)
                file.close()
            else:
                dist_list = json.load(open(filename))
                distributions.append(dist_list)
    else:
        agent_id = create_RAT_agent(agent, source='SFFAN', spreading=True, activation_base=2, decay_parameter=0.05,
                                 constant_offset=0, ngrams=GoogleNGram('~/ngram'), threshold=threshold, var_type=var_type,
                                 discount=discount, bounded=bounded, cooc_depth=cooc_depth, func=func)
        rat = RatTest()
        filename = "agent_distributions/list_" + rat.to_string_id() + "_" + agent_id.to_string_id() + ".json"
        if not os.path.isfile(filename):
            dist = rat.get_rat_agent_distribution(agent_id)
            distributions.append(dist)
            file = open(filename, 'w')
            json.dump(dist, file)
            file.close()
        else:
            dist_list = json.load(open(filename))
            distributions.append(dist_list)
    return distributions

def get_average_integrated_distribution(task, agent, context_type="word", clear="never", whole_corpus=False, threshold=0,
                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1, func='sqrt',
                                        num_points=100):
    num_items_list = []
    solution_loc_list = []
    solution_in_list = []
    if task == "wsd":
        task_id = WSDTask()
        agent_id = create_WSD_agent(agent, partition=1, num_sentences=5000, context_type=context_type, spreading=True, clear=clear,
                 activation_base=2, decay_parameter=0.05, constant_offset=0, whole_corpus=whole_corpus, threshold=threshold,
                 var_type=var_type, discount=discount, activate_answer=False, activate_sentence_words=True, bounded=bounded,
                     num_context_acts=num_context_acts, cooc_depth=cooc_depth, func=func)
    else:
        task_id = RatTest()
        agent_id = create_RAT_agent(agent, source='SFFAN', spreading=True, activation_base=2, decay_parameter=0.05,
                                 constant_offset=0, ngrams=GoogleNGram('~/ngram'), threshold=threshold, var_type=var_type,
                                 discount=discount, bounded=bounded, cooc_depth=cooc_depth, func=func)
    filename = "agent_distributions/list_" + "averaged_" + task_id.to_string_id() + "_" + agent_id.to_string_id() + ".json"
    if not os.path.isfile(filename):
        dists = get_integrated_distribution(task, agent, context_type, clear, whole_corpus, threshold,
                 var_type, discount, bounded, num_context_acts, cooc_depth, func)
        avg_dists = [0] * num_points
        num_dists = 0
        x_vals = np.linspace(0, 1, num_points)
        counter = 0
        for part in dists:  # Getting each distribution
            print(len(part))
            for trial in part:
                print("trial", trial)
                num_dists += 1
                if trial[1] == []:
                    continue
                candidates = trial[1]
                trial_dist_elems, trial_dist_probs = zip(*candidates)
                dist = sorted(trial_dist_probs, reverse=True)
                if trial[0] in trial_dist_elems:
                    solution_in_list.append(1)
                    solution_loc_list.append(dist.index(trial_dist_probs[trial_dist_elems.index(trial[0])]))
                else:
                    solution_in_list.append(0)
                num_items_list.append(len(dist))
                dist_sum = sum(dist)
                if dist_sum > 0:
                    norm_dist = [x/dist_sum for x in dist]
                else:
                    norm_dist = dist
                counter += 1
                if len(norm_dist) == 0:
                    num_dists -= 1
                    continue
                elif len(norm_dist) == 1:
                    norm_dist = [norm_dist[0], norm_dist[0]]
                dist_x_vals = np.linspace(0, 1, len(norm_dist))
                y_interp = interp1d(dist_x_vals, norm_dist)
                for i in range(num_points):
                    avg_dists[i] += y_interp(x_vals[i])
        avg_dist = [list(x_vals), [elem / num_dists for elem in avg_dists]]
        file = open(filename, 'w')
        json.dump(avg_dist, file)
        file.close()
        print("Average Num Items:", sum(num_items_list)/len(num_items_list))
        print("Contains Solution?", sum(solution_in_list)/len(solution_in_list))
        if len(solution_loc_list) > 0:
            print("Solution Location:", sum(solution_loc_list)/len(solution_loc_list))
    else:
        avg_dist = json.load(open(filename))
    return avg_dist

def get_answer_rankings(agent, dists):
    print("agent")
    rankings = []
    counter = 0
    rank_count = 0
    for partition in dists:
        for dist in partition:
            counter += 1
            answer = dist[0]
            if dist[1] == []:
                rankings.append([str(answer), agent, None])
            else:
                candidates, candidate_probs = zip(*dist[1])
                ranked_candidate_probs = sorted(candidate_probs, reverse=True)
                if answer in candidates:
                    answer_index = candidates.index(answer)
                    probs_value = candidate_probs[answer_index]
                    num_probs_vals = candidate_probs.count(probs_value)
                    ranking = ranked_candidate_probs.index(probs_value)
                    if ranking == 0:
                        rank_count += 1/num_probs_vals
                    if num_probs_vals > 1:
                        ranking = (2 * ranking + num_probs_vals) / 2
                    rankings.append([str(answer), agent, ranking])
                else:
                    rankings.append([str(answer), agent, None])
    print(rank_count/counter)
    return rankings

def add_correct_lines(dist, int_dist):
    edited_dist = []
    for index in range(len(dist)):
        dist_to_edit = dist[index]
        if dist_to_edit is None:
            continue
        print(dist_to_edit)
        print("int")
        print(int_dist[index][2])
        if int_dist[index][2] is None:
            dist_to_edit.append("incorrect")
            edited_dist.append(dist_to_edit)
        elif int_dist[index][2] == 0:
            dist_to_edit.append("correct")
            edited_dist.append(dist_to_edit)
        else:
            dist_to_edit.append("incorrect")
            edited_dist.append(dist_to_edit)
    return edited_dist

def integrated_answer_comp_plot(task, agent, show=False):
    adj_cooc_rankings = []
    adj_int_rankings = []
    adj_spread_rankings = []
    points = {}
    if agent == "cts" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                print("cooccurrence")
                cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                "cooccurrence",
                                                                                                clear=clear,
                                                                                                bounded=bounded))
                print("integrated")
                int_rankings = get_answer_rankings(agent,
                                                   get_integrated_distribution(task, "cooc_thresh_sem", clear=clear,
                                                                               bounded=bounded))
                print("spreading")
                spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                               clear=clear,
                                                                                               bounded=bounded))
                adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "cts" and task == "rat":
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "cooc_thresh_sem", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "stc" and task == "wsd":
        for context in ["word", "sense"]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            context_type=context))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "sem_thresh_cooc", context_type=context))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           context_type=context))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "stc" and task == "rat":
        print("cooccurrence")
        cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                        "cooccurrence"))
        print("integrated")
        int_rankings = get_answer_rankings(agent,
                                           get_integrated_distribution(task, "sem_thresh_cooc"))
        print("spreading")
        spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading"))
        adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
        adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
        adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "joint" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                    print("cooccurrence")
                    cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                    "cooccurrence",
                                                                                                    context_type=context,
                                                                                                    clear=clear,
                                                                                                    bounded=bounded))
                    print("integrated")
                    int_rankings = get_answer_rankings(agent,
                                                       get_integrated_distribution(task, "joint_prob", context_type=context, clear=clear, bounded=bounded))
                    print("spreading")
                    spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                                   context_type=context, clear=clear,
                                                                                                   bounded=bounded))
                    adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                    adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                    adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "joint" and task == "rat":
        print("yes")
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings("joint",
                                               get_integrated_distribution(task, "joint_prob", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("spreading", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "add" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("cooccurrence")
                    cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                    "cooccurrence",
                                                                                                    clear=clear,
                                                                                                    context_type=context,
                                                                                                    bounded=bounded))
                    print("integrated")
                    int_rankings = get_answer_rankings(agent,
                                                       get_integrated_distribution(task, "add_prob", clear=clear,
                                                                                                    context_type=context,
                                                                                                    bounded=bounded))
                    print("spreading")
                    spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                                   clear=clear,
                                                                                                   context_type=context,
                                                                                                   bounded=bounded))
                    adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                    adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                    adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "add" and task == "rat":
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "add_prob", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "sbc" and task == "wsd":
        for context in ["word", "sense"]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            context_type=context))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "spreading_boosted_cooc", context_type=context))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           context_type=context))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "sbc" and task == "rat":
        print("cooccurrence")
        cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                        "cooccurrence"))
        print("integrated")
        int_rankings = get_answer_rankings(agent,
                                           get_integrated_distribution(task, "spreading_boosted_cooc"))
        print("spreading")
        spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading"))
        adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
        adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
        adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == 'var' and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    for vartype in ["maxdiff", "stdev"]:
                        print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                        print("cooccurrence")
                        cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                        "cooccurrence",
                                                                                                        context_type=context,
                                                                                                        clear=clear, var_type=vartype,
                                                                                                        bounded=bounded))
                        print("integrated")
                        int_rankings = get_answer_rankings(agent,
                                                           get_integrated_distribution(task, "joint_var", context_type=context,
                                                                                       clear=clear, var_type=vartype,
                                                                                       bounded=bounded))
                        print("spreading")
                        spread_rankings = get_answer_rankings("semantics",
                                                              get_integrated_distribution(task, "spreading",
                                                                                          context_type=context, clear=clear,
                                                                                          var_type=vartype, bounded=bounded))
                        adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                        adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                        adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "var" and task == "rat":
        for bounded in [True, False]:
            for vartype in ["maxdiff", "stdev"]:
                print("cooccurrence")
                cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                "cooccurrence",
                                                                                                var_type=vartype,
                                                                                                bounded=bounded))
                print("integrated")
                int_rankings = get_answer_rankings(agent,
                                                   get_integrated_distribution(task, "joint_var", var_type=vartype, bounded=bounded))
                print("spreading")
                spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                               var_type=vartype, bounded=bounded))
                adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "max" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                    print("cooccurrence")
                    cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                    "cooccurrence",
                                                                                                    context_type=context, clear=clear,
                                                                                                    bounded=bounded))
                    print("integrated")
                    int_rankings = get_answer_rankings(agent,
                                                       get_integrated_distribution(task, "max_prob", context_type=context, clear=clear,
                                                                                   bounded=bounded))
                    print("spreading")
                    spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                                   context_type=context, clear=clear,
                                                                                                   bounded=bounded))
                    adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                    adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                    adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "max" and task == "rat":
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "max_prob", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "cws" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                    print("cooccurrence")
                    cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                    "cooccurrence",
                                                                                                    context_type=context, clear=clear,
                                                                                                    bounded=bounded))
                    print("integrated")
                    int_rankings = get_answer_rankings(agent,
                                                       get_integrated_distribution(task, "cooc_weight_spreading", context_type=context, clear=clear,
                                                                                   bounded=bounded))
                    print("spreading")
                    spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                                   context_type=context, clear=clear,
                                                                                                   bounded=bounded))
                    adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                    adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                    adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "cws" and task == "rat":
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "cooc_weight_spreading", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "ssc" and task == "wsd":
        for context in ["word", "sense"]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            context_type=context))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "spread_supplemented_cooc", context_type=context))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           context_type=context))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "ssc" and task == "rat":
        print("cooccurrence")
        cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                        "cooccurrence"))
        print("integrated")
        int_rankings = get_answer_rankings(agent,
                                           get_integrated_distribution(task, "spread_supplemented_cooc"))
        print("spreading")
        spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading"))
        adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
        adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
        adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "css" and task == "wsd":
        for clear in ["word", "never", "sentence"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                    print("cooccurrence")
                    cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                    "cooccurrence",
                                                                                                    context_type=context, clear=clear,
                                                                                                    bounded=bounded))
                    print("integrated")
                    int_rankings = get_answer_rankings(agent,
                                                       get_integrated_distribution(task, "cooc_supplemented_spreading", context_type=context, clear=clear,
                                                                                                    bounded=bounded))
                    print("spreading")
                    spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                                   context_type=context,
                                                                                                   clear=clear,
                                                                                                   bounded=bounded))
                    adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                    adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                    adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "css" and task == "rat":
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "cooc_supplemented_spreading", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "ces" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                    print("cooccurrence")
                    cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                                    "cooccurrence",
                                                                                                    context_type=context, clear=clear,
                                                                                                    bounded=bounded))
                    print("integrated")
                    int_rankings = get_answer_rankings(agent,
                                                       get_integrated_distribution(task, "cooc_expanded_spreading", context_type=context, clear=clear,
                                                                                                    bounded=bounded))
                    print("spreading")
                    spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                                   context_type=context,
                                                                                                   clear=clear,
                                                                                                   bounded=bounded))
                    adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
                    adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
                    adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    elif agent == "ces" and task == "rat":
        for bounded in [True, False]:
            print("cooccurrence")
            cooc_rankings = get_answer_rankings("cooccurrence", get_integrated_distribution(task,
                                                                                            "cooccurrence",
                                                                                            bounded=bounded))
            print("integrated")
            int_rankings = get_answer_rankings(agent,
                                               get_integrated_distribution(task, "cooc_expanded_spreading", bounded=bounded))
            print("spreading")
            spread_rankings = get_answer_rankings("semantics", get_integrated_distribution(task, "spreading",
                                                                                           bounded=bounded))
            adj_cooc_rankings.extend(add_correct_lines(dist=cooc_rankings, int_dist=int_rankings))
            adj_int_rankings.extend(add_correct_lines(dist=int_rankings, int_dist=int_rankings))
            adj_spread_rankings.extend(add_correct_lines(dist=spread_rankings, int_dist=int_rankings))
    for i in range(len(adj_cooc_rankings)):
        ranks = tuple([adj_cooc_rankings[i][2], adj_int_rankings[i][2], adj_spread_rankings[i][2]])
        if ranks in points:
            points[ranks] += 1
        else:
            points[ranks] = 1
    fig, ax = plt.subplots()
    # for i in range(len(adj_cooc_rankings)):
    #     if adj_int_rankings[i][3] == "correct":
    #         color = "green"
    #     else:
    #         color = "red"
    #     ax.plot([0.5, 1.5, 2.5], , color=color)
    print(len(points))
    num_points = sum(points.values())
    print(num_points)
    green_list = []
    red_list = []
    for point in points:
        if point[1] == 0:
            # Green
            green_list.append(point)
        else:
            # Red
            red_list.append(point)
    for point in red_list:
        alpha = (points[point] / num_points) * (4 / 5) + (1 / 5)
        ax.plot([0.5, 1.5, 2.5], point, color="red", alpha=alpha)
    for point in green_list:
        alpha = (points[point] / num_points) * (4/5) + (1/5)
        ax.plot([0.5, 1.5, 2.5], point, color="green", alpha=alpha)
    ax.set_xticks([0.5, 1.5, 2.5], ["cooccurrence", agent, "semantics"])
    ax.set_title(task)
    plt.savefig(task + "_" + agent + ".png")
    if show:
        plt.show()

def get_probs_from_trial(trial, sem=False, task = "wsd"):
    candidates, candidate_probs = zip(*trial[1])
    # Normalizing...
    if sem:
        if any([prob > 150 for prob in candidate_probs]):
            return []
        # Calculate semantic probabilities...
        if task == "wsd":
            tau = -float("inf")
        else:
            tau = math.log(3/8)
        normalized_candidate_probs = []
        for prob in candidate_probs:
            num = math.exp(prob/ 0.25)
            denom = math.exp(tau / 0.25) + sum(math.exp(act / 0.25) for act in candidate_probs)
            normalized_candidate_probs.append(num/denom)
        prob_sum = sum(normalized_candidate_probs)
        if prob_sum != 0:
            normalized_candidate_probs = [float(prob/prob_sum) for prob in normalized_candidate_probs]
    else:
        prob_sum = sum(candidate_probs)
        if prob_sum != 0:
            normalized_candidate_probs = [prob/prob_sum for prob in candidate_probs]
        else:
            normalized_candidate_probs = candidate_probs
    # Finding answer prob
    return normalized_candidate_probs

def get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False, task="wsd"):
    sem_ranking_dists = get_answer_rankings("semantic", sem_dists)
    cooc_ranking_dists = get_answer_rankings("cooc", cooc_dists)
    int_ranking_dists = get_answer_rankings("int", int_dists)
    mtx_dict = {"sem_up_cooc_up": 0,
                   "sem_up_cooc_eq": 0,
                   "sem_up_cooc_down": 0,
                   "sem_eq_cooc_up": 0,
                   "sem_eq_cooc_eq": 0,
                   "sem_eq_cooc_down": 0,
                   "sem_down_cooc_up": 0,
                   "sem_down_cooc_eq": 0,
                   "sem_down_cooc_down": 0}
    if len(sem_ranking_dists) != len(int_ranking_dists) or len(cooc_ranking_dists) != len(int_ranking_dists):
        raise ValueError()
    for index in range(len(sem_ranking_dists)):
        sem_rank = sem_ranking_dists[index][2]
        #print("sem_rank", sem_rank)
        cooc_rank = cooc_ranking_dists[index][2]
        #print("cooc_rank", cooc_rank)
        int_rank = int_ranking_dists[index][2]
        #print("int_rank", int_rank)
        mtx_cat = ""
        if int_rank is None:
            mtx_cat += "sem_down_cooc_down"
        else:
            if sem_rank is None:
                mtx_cat += "sem_up_"
            elif int_rank == sem_rank:
                mtx_cat += "sem_eq_"
            elif int_rank < sem_rank:
                mtx_cat += "sem_up_"
            else:
                mtx_cat += "sem_down_"
            if cooc_rank is None:
                mtx_cat += "cooc_up"
            elif int_rank == cooc_rank:
                mtx_cat += "cooc_eq"
            elif int_rank < cooc_rank:
                mtx_cat += "cooc_up"
            else:
                mtx_cat += "cooc_down"
        mtx_dict[mtx_cat] += 1
        #print("label", mtx_cat)
    return mtx_dict

def update_mtx_dict(dist_mtx_dict, trial_mtx_dict):
    dist_mtx_dict["sem_up_cooc_up"] += trial_mtx_dict["sem_up_cooc_up"]
    dist_mtx_dict["sem_up_cooc_eq"] += trial_mtx_dict["sem_up_cooc_eq"]
    dist_mtx_dict["sem_up_cooc_down"] += trial_mtx_dict["sem_up_cooc_down"]
    dist_mtx_dict["sem_eq_cooc_up"] += trial_mtx_dict["sem_eq_cooc_up"]
    dist_mtx_dict["sem_eq_cooc_eq"] += trial_mtx_dict["sem_eq_cooc_eq"]
    dist_mtx_dict["sem_eq_cooc_down"] += trial_mtx_dict["sem_eq_cooc_down"]
    dist_mtx_dict["sem_down_cooc_up"] += trial_mtx_dict["sem_down_cooc_up"]
    dist_mtx_dict["sem_down_cooc_eq"] += trial_mtx_dict["sem_down_cooc_eq"]
    dist_mtx_dict["sem_down_cooc_down"] += trial_mtx_dict["sem_down_cooc_down"]
    return dist_mtx_dict

def get_agent_matrices(agent, task):
    print("agent", agent, "task", task)
    mtx_dict = {"sem_up_cooc_up": 0,
                "sem_up_cooc_eq": 0,
                "sem_up_cooc_down": 0,
                "sem_eq_cooc_up": 0,
                "sem_eq_cooc_eq": 0,
                "sem_eq_cooc_down": 0,
                "sem_down_cooc_up": 0,
                "sem_down_cooc_eq": 0,
                "sem_down_cooc_down": 0}

    if agent == "cts" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                int_dists = get_integrated_distribution("wsd", "cooc_thresh_sem", "word", clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                cooc_dists = get_integrated_distribution("wsd", "cooccurrence", "word", clear="never", whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1, func='sqrt')
                sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                mtx_dict = update_mtx_dict(mtx_dict, get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True, task=task))

    elif agent == "cts" and task == "rat":
        for bounded in [True, False]:
            int_dists = get_integrated_distribution("rat", "cooc_thresh_sem", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))

    elif agent == "stc" and task == "wsd":
        for context in ["word", "sense"]:
            int_dists = get_integrated_distribution("wsd", "sem_thresh_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1, func='sqrt')
            cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "stc" and task == "rat":
        print("stc, rat")
        int_dists = get_integrated_distribution("rat", "sem_thresh_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt')
        cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1, func='sqrt')
        sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        mtx_dict = update_mtx_dict(mtx_dict,
                                   get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False, task=task))

    elif agent == "joint" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "joint_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                         task=task))

    elif agent == "joint" and task == "rat":
        for bounded in [True, False]:
            int_dists = get_integrated_distribution("rat", "joint_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "add" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("add, wsd, ", clear, ",",context, ",", bounded)
                    int_dists= get_integrated_distribution("wsd", "add_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                         task=task))

    elif agent == "add" and task == "rat":
        for bounded in [True, False]:
            int_dists = get_integrated_distribution("rat", "add_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "sbc" and task == "wsd":
        for context in ["word", "sense"]:
            print("sbc, wsd,", context)
            int_dists = get_integrated_distribution("wsd", "spreading_boosted_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "sbc" and task == "rat":
        int_dists = get_integrated_distribution("rat", "spreading_boosted_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt')
        cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1, func='sqrt')
        sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        mtx_dict = update_mtx_dict(mtx_dict,
                                   get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                             task=task))

    elif agent == 'var' and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    for vartype in ["maxdiff", "stdev"]:
                        int_dists = get_integrated_distribution("wsd", "joint_var", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type=vartype, discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                        cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                                 whole_corpus=False, threshold=0,
                                                                 var_type="stdev", discount=0.1, bounded=False,
                                                                 num_context_acts=1, cooc_depth=1, func='sqrt')
                        sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear,
                                                                whole_corpus=False, threshold=0,
                                                                var_type="stdev", discount=0.1, bounded=bounded,
                                                                num_context_acts=1, cooc_depth=1,
                                                                func='sqrt')
                        mtx_dict = update_mtx_dict(mtx_dict, get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                                       int_sem=False, task=task))

    elif agent == "var" and task == "rat":
        for bounded in [True, False]:
            for vartype in ["maxdiff", "stdev"]:
                int_dists = get_integrated_distribution("rat", "joint_var", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type=vartype, discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
                cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never",
                                                         whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=False,
                                                         num_context_acts=1,
                                                         cooc_depth=1, func='sqrt')
                sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                        threshold=0,
                                                        var_type="stdev", discount=0.1, bounded=bounded,
                                                        num_context_acts=1,
                                                        cooc_depth=1,
                                                        func='sqrt')
                mtx_dict = update_mtx_dict(mtx_dict,
                                           get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                     task=task))

    elif agent == "max" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("max, wsd, ", clear, ",",context, ",", bounded)
                    int_dists = get_integrated_distribution("wsd", "max_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                         task=task))

    elif agent == "max" and task == "rat":
        for bounded in [True, False]:
            print("max, rat", bounded)
            int_dists = get_integrated_distribution("rat", "max_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "cws" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "cooc_weight_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                         int_sem=True,
                                                                         task=task))

    elif agent == "cws" and task == "rat":
        for bounded in [True, False]:
            print("cws, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_weight_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))

    elif agent == "ssc" and task == "wsd":
        for context in ["word", "sense"]:
            print("ssc, wsd,", context)
            int_dists = get_integrated_distribution("wsd", "spread_supplemented_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                     whole_corpus=False, threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False,
                                                     num_context_acts=1, cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False,
                                                    num_context_acts=1, cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "ssc" and task == "rat":
        print("ssc, rat")
        int_dists = get_integrated_distribution("rat", "spread_supplemented_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt')
        cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1, func='sqrt')
        sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        mtx_dict = update_mtx_dict(mtx_dict,
                                   get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                             task=task))

    elif agent == "css" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "cooc_supplemented_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                         int_sem=True,
                                                                         task=task))

    elif agent == "css" and task == "rat":
        for bounded in [True, False]:
            print("css, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_supplemented_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))

    elif agent == "ces" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "cooc_expanded_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                         int_sem=True,
                                                                         task=task))

    elif agent == "ces" and task == "rat":
        for bounded in [True, False]:
            print("ces, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_expanded_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))
    mtx_sum = sum(mtx_dict.values())
    if mtx_sum != 0:
        mtx_dict = {k: v/mtx_sum for k,v in mtx_dict.items()}
    print(mtx_dict.items())
    return mtx_dict

def get_candidates_preserved(agent, task):
    print("agent", agent, "task", task)
    mtx_dict = {"sem_up_cooc_up": 0,
                "sem_up_cooc_eq": 0,
                "sem_up_cooc_down": 0,
                "sem_eq_cooc_up": 0,
                "sem_eq_cooc_eq": 0,
                "sem_eq_cooc_down": 0,
                "sem_down_cooc_up": 0,
                "sem_down_cooc_eq": 0,
                "sem_down_cooc_down": 0}

    if agent == "cts" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                int_dists = get_integrated_distribution("wsd", "cooc_thresh_sem", "word", clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                cooc_dists = get_integrated_distribution("wsd", "cooccurrence", "word", clear="never", whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1, func='sqrt')
                sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                mtx_dict = update_mtx_dict(mtx_dict, get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True, task=task))

    elif agent == "cts" and task == "rat":
        for bounded in [True, False]:
            print("cts, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_thresh_sem", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))

    elif agent == "stc" and task == "wsd":
        for context in ["word", "sense"]:
            print("stc, wsd,", context)
            int_dists = get_integrated_distribution("wsd", "sem_thresh_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1, func='sqrt')
            cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "stc" and task == "rat":
        print("stc, rat")
        int_dists = get_integrated_distribution("rat", "sem_thresh_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt')
        cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1, func='sqrt')
        sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        mtx_dict = update_mtx_dict(mtx_dict,
                                   get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False, task=task))

    elif agent == "joint" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "joint_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                         task=task))

    elif agent == "joint" and task == "rat":
        for bounded in [True, False]:
            int_dists = get_integrated_distribution("rat", "joint_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "add" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("add, wsd, ", clear, ",",context, ",", bounded)
                    int_dists= get_integrated_distribution("wsd", "add_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                         task=task))

    elif agent == "add" and task == "rat":
        for bounded in [True, False]:
            int_dists = get_integrated_distribution("rat", "add_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "sbc" and task == "wsd":
        for context in ["word", "sense"]:
            print("sbc, wsd,", context)
            int_dists = get_integrated_distribution("wsd", "spreading_boosted_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "sbc" and task == "rat":
        int_dists = get_integrated_distribution("rat", "spreading_boosted_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt')
        cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1, func='sqrt')
        sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        mtx_dict = update_mtx_dict(mtx_dict,
                                   get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                             task=task))

    elif agent == 'var' and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    for vartype in ["maxdiff", "stdev"]:
                        int_dists = get_integrated_distribution("wsd", "joint_var", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type=vartype, discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                        cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                                 whole_corpus=False, threshold=0,
                                                                 var_type="stdev", discount=0.1, bounded=False,
                                                                 num_context_acts=1, cooc_depth=1, func='sqrt')
                        sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear,
                                                                whole_corpus=False, threshold=0,
                                                                var_type="stdev", discount=0.1, bounded=bounded,
                                                                num_context_acts=1, cooc_depth=1,
                                                                func='sqrt')
                        mtx_dict = update_mtx_dict(mtx_dict, get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                                       int_sem=False, task=task))

    elif agent == "var" and task == "rat":
        for bounded in [True, False]:
            for vartype in ["maxdiff", "stdev"]:
                int_dists = get_integrated_distribution("rat", "joint_var", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type=vartype, discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
                cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never",
                                                         whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=False,
                                                         num_context_acts=1,
                                                         cooc_depth=1, func='sqrt')
                sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                        threshold=0,
                                                        var_type="stdev", discount=0.1, bounded=bounded,
                                                        num_context_acts=1,
                                                        cooc_depth=1,
                                                        func='sqrt')
                mtx_dict = update_mtx_dict(mtx_dict,
                                           get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                     task=task))

    elif agent == "max" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("max, wsd, ", clear, ",",context, ",", bounded)
                    int_dists = get_integrated_distribution("wsd", "max_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                         task=task))

    elif agent == "max" and task == "rat":
        for bounded in [True, False]:
            print("max, rat", bounded)
            int_dists = get_integrated_distribution("rat", "max_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "cws" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "cooc_weight_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                         int_sem=True,
                                                                         task=task))

    elif agent == "cws" and task == "rat":
        for bounded in [True, False]:
            print("cws, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_weight_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))

    elif agent == "ssc" and task == "wsd":
        for context in ["word", "sense"]:
            print("ssc, wsd,", context)
            int_dists = get_integrated_distribution("wsd", "spread_supplemented_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                     whole_corpus=False, threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False,
                                                     num_context_acts=1, cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False,
                                                    num_context_acts=1, cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                                 task=task))

    elif agent == "ssc" and task == "rat":
        print("ssc, rat")
        int_dists = get_integrated_distribution("rat", "spread_supplemented_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt')
        cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1, func='sqrt')
        sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        mtx_dict = update_mtx_dict(mtx_dict,
                                   get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=False,
                                                             task=task))

    elif agent == "css" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "cooc_supplemented_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                         int_sem=True,
                                                                         task=task))

    elif agent == "css" and task == "rat":
        for bounded in [True, False]:
            print("css, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_supplemented_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))

    elif agent == "ces" and task == "wsd":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    int_dists = get_integrated_distribution("wsd", "cooc_expanded_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt')
                    cooc_dists = get_integrated_distribution("wsd", "cooccurrence", context, clear="never",
                                                             whole_corpus=False, threshold=0,
                                                             var_type="stdev", discount=0.1, bounded=False,
                                                             num_context_acts=1, cooc_depth=1, func='sqrt')
                    sem_dists = get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')
                    mtx_dict = update_mtx_dict(mtx_dict,
                                               get_distribution_matrices(sem_dists, cooc_dists, int_dists,
                                                                         int_sem=True,
                                                                         task=task))

    elif agent == "ces" and task == "rat":
        for bounded in [True, False]:
            print("ces, rat", bounded)
            int_dists = get_integrated_distribution("rat", "cooc_expanded_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')
            cooc_dists = get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                     cooc_depth=1, func='sqrt')
            sem_dists = get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')
            mtx_dict = update_mtx_dict(mtx_dict,
                                       get_distribution_matrices(sem_dists, cooc_dists, int_dists, int_sem=True,
                                                                 task=task))
    mtx_sum = sum(mtx_dict.values())
    percent_mtx_dict = {k: v/mtx_sum for k,v in mtx_dict.items()}
    print(percent_mtx_dict.items())
    return percent_mtx_dict

def get_dist_stats(dists, sem=False):
    avg_sum = 0
    var_sum = 0
    num_dists = 0
    decile_sum = np.zeros(10)
    min_sum = 0
    max_sum = 0
    for partition in dists:
        for trial in partition:
            probs = sorted(get_probs_from_trial(trial, sem))
            if probs == []:
                continue
            avg_sum += sum(probs)/len(probs)
            var_sum += np.var(probs)
            decile_sum += np.percentile(np.array(probs), np.arange(0, 100, 10))
            min_sum += min(probs)
            max_sum += max(probs)
            num_dists += 1
        print(decile_sum)
    if num_dists > 0:
        print("Average Mean:", avg_sum / num_dists)
        print("Average Variance:", var_sum / num_dists)
        print("Average Deciles:", decile_sum / num_dists)
        print("Average Min:", min_sum / num_dists)
        print("Average Max:", max_sum / num_dists)
    else:
        print("Zero!")
    return

def get_agent_stats(agent, task):
    if agent == "cooccurrence" and task == "wsd":
        print("cooccurrence wsd context: sense")
        dists = []
        dists.extend(get_integrated_distribution("wsd", "cooccurrence", "sense", clear="never", whole_corpus=False,
                                                   threshold=0,
                                                   var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                   cooc_depth=1, func='sqrt'))
        print("cooccurrence wsd context: word")
        dists.extend(get_integrated_distribution("wsd", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                   threshold=0,
                                                   var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                   cooc_depth=1,
                                                   func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "cooccurrence" and task == "rat":
        print("cooccurrence rat")
        dists = []
        dists.extend(get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                   threshold=0,
                                                   var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                   cooc_depth=1,
                                                   func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "semantics" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                print("spreading wsd clear:", clear, "bounded:", bounded)
                dists.extend(get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                           threshold=0,
                                                           var_type="stdev", discount=0.1, bounded=bounded,
                                                           num_context_acts=1, cooc_depth=1,
                                                           func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "semantics" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("spreading rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                       threshold=0,
                                                       var_type="stdev", discount=0.1, bounded=bounded,
                                                       num_context_acts=1,
                                                       cooc_depth=1,
                                                       func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "cts" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                print("cts wsd clear:", clear, "bounded:", bounded)
                dists.extend(get_integrated_distribution("wsd", "cooc_thresh_sem", "word", clear=clear,
                                                        whole_corpus=False, threshold=0,
                                                        var_type="stdev", discount=0.1, bounded=bounded,
                                                        num_context_acts=1, cooc_depth=1,
                                                        func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "cts" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("cts rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "cooc_thresh_sem", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "stc" and task == "wsd":
        dists = []
        for context in ["word", "sense"]:
            print("stc wsd context:", context)
            dists.extend(get_integrated_distribution("wsd", "sem_thresh_cooc", context, clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1, func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "stc" and task == "rat":
        print("stc rat")
        dists = []
        dists.extend(get_integrated_distribution("rat", "sem_thresh_cooc", "word", clear="never", whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "joint" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                    dists.extend(get_integrated_distribution("wsd", "joint_prob", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "joint" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("joint rat", "bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "joint_prob", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "add" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("add wsd clear:", clear, "context:", context, "bounded:", bounded)
                    dists.extend(get_integrated_distribution("wsd", "add_prob", context, clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "add" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("add rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "add_prob", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "sbc" and task == "wsd":
        dists = []
        for context in ["word", "sense"]:
            print("sbc wsd context:", context)
            dists.extend(get_integrated_distribution("wsd", "spreading_boosted_cooc", context, clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "sbc" and task == "rat":
        print("sbc rat")
        dists = []
        dists.extend(get_integrated_distribution("rat", "spreading_boosted_cooc", "word", clear="never",
                                                whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == 'var' and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    for vartype in ["maxdiff", "stdev"]:
                        print("var wsd clear:", clear, "bounded:", bounded, "context:", context, "vartype:", vartype)
                        dists.extend(get_integrated_distribution("wsd", "joint_var", context, clear=clear,
                                                                whole_corpus=False, threshold=0,
                                                                var_type=vartype, discount=0.1, bounded=bounded,
                                                                num_context_acts=1, cooc_depth=1,
                                                                func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "var" and task == "rat":
        dists = []
        for bounded in [True, False]:
            for vartype in ["maxdiff", "stdev"]:
                print("var rat", "bounded:", bounded, "vartype:", vartype)
                dists.extend(get_integrated_distribution("rat", "joint_var", "word", clear="never", whole_corpus=False,
                                                        threshold=0,
                                                        var_type=vartype, discount=0.1, bounded=bounded,
                                                        num_context_acts=1,
                                                        cooc_depth=1,
                                                        func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "max" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("max wsd clear:", clear, "context:", context, "bounded:", bounded)
                    dists.extend(get_integrated_distribution("wsd", "max_prob", context, clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "max" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("max rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "max_prob", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "cws" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("cws wsd clear:", clear, "bounded:", bounded, "context:", context)
                    dists.extend(get_integrated_distribution("wsd", "cooc_weight_spreading", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "cws" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("cws rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "cooc_weight_spreading", "word", clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "ssc" and task == "wsd":
        dists = []
        for context in ["word", "sense"]:
            print("ssc wsd context:", context)
            dists.extend(get_integrated_distribution("wsd", "spread_supplemented_cooc", context, clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=False)
    elif agent == "ssc" and task == "rat":
        print("ssc rat")
        dist = get_integrated_distribution("rat", "spread_supplemented_cooc", "word", clear="never",
                                                whole_corpus=False,
                                                threshold=0,
                                                var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                cooc_depth=1,
                                                func='sqrt')
        get_dist_stats(dist, sem=False)
    elif agent == "css" and task == "wsd":
        dists = []
        for clear in ["word", "never", "sentence"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("css wsd clear:", clear, "bounded:", bounded, "context:", context)
                    dists.extend(get_integrated_distribution("wsd", "cooc_supplemented_spreading", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "css" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("css rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "cooc_supplemented_spreading", "word", clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "ces" and task == "wsd":
        dists = []
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("ces wsd clear:", clear, "bounded:", bounded, "context:", context)
                    dists.extend(get_integrated_distribution("wsd", "cooc_expanded_spreading", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt'))
        get_dist_stats(dists, sem=True)
    elif agent == "ces" and task == "rat":
        dists = []
        for bounded in [True, False]:
            print("ces rat bounded:", bounded)
            dists.extend(get_integrated_distribution("rat", "cooc_expanded_spreading", "word", clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt'))
        get_dist_stats(dists, sem=True)

def pareto_filter(wsd, rat):
    combined = [[wsd[index], rat[index]] for index in range(len(wsd))]
    pareto_wsd = []
    pareto_rat = []
    if len(combined) <= 1:
        return wsd, rat
    for point in combined:
        is_dominated = False
        for other_point in combined:
            if point[0] < other_point[0] and point[1] <= other_point[1]:
                print("yes")
                is_dominated = True
                break
        if not is_dominated:
            pareto_wsd.append(point[0])
            pareto_rat.append(point[1])
    return pareto_wsd, pareto_rat

def get_discounted_accuracies(dists):
    cumulative_acc_sum = 0
    total_num_trials = 0
    for partition in dists:
        for trial in partition:
            total_num_trials += 1
            answer = trial[0]
            if trial[1] == []:
                continue
            candidates, candidate_probs = zip(*trial[1])
            ordered_candidate_probs = sorted(candidate_probs, reverse=True)
            if answer in candidates:
                answer_index = candidates.index(answer)
                if candidate_probs[answer_index] == ordered_candidate_probs[0]:
                    num_max_candidates = candidate_probs.count(ordered_candidate_probs[0])
                    cumulative_acc_sum += 1/num_max_candidates
    return cumulative_acc_sum / total_num_trials

def pareto_plot():
    fig, ax = plt.subplots()
    ax.set_title("RAT vs. WSD Accuracy for Retrieval Mechanisms")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([-0.01, 0.45])
    ax.set_ylabel("RAT Accuracy")
    ax.set_xlabel("WSD Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            print("cts wsd clear:", clear, "bounded:", bounded)
            wsd.append(
                get_discounted_accuracies(get_integrated_distribution("wsd", "cooc_thresh_sem", "word", clear=clear,
                                                                      whole_corpus=False, threshold=0,
                                                                      var_type="stdev", discount=0.1, bounded=bounded,
                                                                      num_context_acts=1, cooc_depth=1,
                                                                      func='sqrt')))
            rat.append(get_discounted_accuracies(
                get_integrated_distribution("rat", "cooc_thresh_sem", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="tab:blue", label="CTS")

    print("cooccurrence")
    wsd = []
    rat = []
    for context in ["sense", "word"]:
        wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "cooccurrence", context, clear="never", whole_corpus=False,
                                                   threshold=0,
                                                   var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                   cooc_depth=1, func='sqrt')))
        rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False,
                                                   threshold=0,
                                                   var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                   cooc_depth=1,
                                                   func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="black", label="Co-occurrence")

    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            print("spreading wsd clear:", clear, "bounded:", bounded)
            wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False,
                                                           threshold=0,
                                                           var_type="stdev", discount=0.1, bounded=bounded,
                                                           num_context_acts=1, cooc_depth=1,
                                                           func='sqrt')))
            rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                                     threshold=0,
                                                     var_type="stdev", discount=0.1, bounded=bounded,
                                                     num_context_acts=1,
                                                     cooc_depth=1,
                                                     func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="black", label="Semantic Spreading")



    wsd = []
    rat = []
    for context in ["word", "sense"]:
        print("stc wsd context:", context)
        wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "sem_thresh_cooc", context, clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1, func='sqrt')))
        rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "sem_thresh_cooc", "word", clear="never", whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1,
                                                 func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="b", label="STC")


    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                print("joint wsd", "clear:", clear, "bounded:", bounded, "context:", context)
                wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "joint_prob", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')))
                rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "joint_prob", "word", clear="never", whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=bounded,
                                                         num_context_acts=1,
                                                         cooc_depth=1,
                                                         func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="red", label="Joint")


    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                print("add wsd clear:", clear, "context:", context, "bounded:", bounded)
                wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "add_prob", context, clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')))
                rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "add_prob", "word", clear="never", whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=bounded,
                                                         num_context_acts=1,
                                                         cooc_depth=1,
                                                         func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="tomato", label="Additive")


    wsd = []
    rat = []
    for context in ["word", "sense"]:
        print("sbc wsd context:", context)
        wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "spreading_boosted_cooc", context, clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')))
        rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "spreading_boosted_cooc", "word", clear="never",
                                                 whole_corpus=False,
                                                 threshold=0,
                                                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                 cooc_depth=1,
                                                 func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="lime", label="SBC")


    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                for vartype in ["maxdiff", "stdev"]:
                    print("var wsd clear:", clear, "bounded:", bounded, "context:", context, "vartype:", vartype)
                    wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "joint_var", context, clear=clear,
                                                                whole_corpus=False, threshold=0,
                                                                var_type=vartype, discount=0.1, bounded=bounded,
                                                                num_context_acts=1, cooc_depth=1,
                                                                func='sqrt')))
                    rat.append(get_discounted_accuracies(
                        get_integrated_distribution("rat", "joint_var", "word", clear="never", whole_corpus=False,
                                                    threshold=0,
                                                    var_type=vartype, discount=0.1, bounded=bounded,
                                                    num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="goldenrod", label="Variance")



    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                print("max wsd clear:", clear, "context:", context, "bounded:", bounded)
                wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "max_prob", context, clear=clear, whole_corpus=False,
                                                            threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')))
                rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "max_prob", "word", clear="never", whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=bounded,
                                                         num_context_acts=1,
                                                         cooc_depth=1,
                                                         func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="darkgoldenrod", label="Max Probability")


    wsd = []
    rat = []
    for context in ["word", "sense"]:
        print("ssc wsd context:", context)
        wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "spread_supplemented_cooc", context, clear="never",
                                                    whole_corpus=False,
                                                    threshold=0,
                                                    var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                                    cooc_depth=1,
                                                    func='sqrt')))
        rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "spread_supplemented_cooc", "word", clear="never",
                                           whole_corpus=False,
                                           threshold=0,
                                           var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                           cooc_depth=1,
                                           func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="tab:purple", label="SSC")


    wsd = []
    rat = []
    for clear in ["word", "never", "sentence"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                print("css wsd clear:", clear, "bounded:", bounded, "context:", context)
                wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "cooc_supplemented_spreading", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')))
                rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "cooc_supplemented_spreading", "word", clear="never",
                                                         whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=bounded,
                                                         num_context_acts=1,
                                                         cooc_depth=1,
                                                         func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="purple", label="CSS")

    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                print("cws wsd clear:", clear, "bounded:", bounded, "context:", context)
                wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "cooc_weight_spreading", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')))
                rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "cooc_weight_spreading", "word", clear="never",
                                                         whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=bounded,
                                                         num_context_acts=1,
                                                         cooc_depth=1,
                                                         func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    print(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="tab:green", label="CWS")

    wsd = []
    rat = []
    for clear in ["never", "sentence", "word"]:
        for bounded in [True, False]:
            for context in ["word", "sense"]:
                print("ces wsd clear:", clear, "bounded:", bounded, "context:", context)
                wsd.append(get_discounted_accuracies(get_integrated_distribution("wsd", "cooc_expanded_spreading", context, clear=clear,
                                                            whole_corpus=False, threshold=0,
                                                            var_type="stdev", discount=0.1, bounded=bounded,
                                                            num_context_acts=1, cooc_depth=1,
                                                            func='sqrt')))
                rat.append(get_discounted_accuracies(get_integrated_distribution("rat", "cooc_expanded_spreading", "word", clear="never",
                                                         whole_corpus=False,
                                                         threshold=0,
                                                         var_type="stdev", discount=0.1, bounded=bounded,
                                                         num_context_acts=1,
                                                         cooc_depth=1,
                                                         func='sqrt')))
    wsd, rat = pareto_filter(wsd, rat)
    ax.scatter(wsd, rat, marker='o', color="cyan", label="CES")

    plt.show()

def dist_v_guesses_comp_wsd(agent, context_type, clear="never", whole_corpus=False, threshold=0, var_type="stdev",
                        discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1, func='sqrt'):
    distributions = get_integrated_distribution("wsd", agent, context_type, clear, whole_corpus, threshold,
                 var_type, discount, bounded, num_context_acts, cooc_depth, func)
    print(len(distributions))
    guesses_files = get_wsd_filenames(agent, context_type=context_type, clear=clear, whole_corpus=whole_corpus,
                                      threshold=threshold, var_type=var_type, discount=discount, bounded=bounded,
                                      num_context_acts=num_context_acts, cooc_depth=cooc_depth, func=func)
    guess_lists = []
    for file in guesses_files:
        guess_lists.append(json.load(open(file)))
    num_errors = 0
    num_trials = 0
    for partition_index in range(6):
        dist_part = distributions[partition_index]
        guess_part = guess_lists[partition_index]
        for trial_index in range(len(dist_part)):
            num_trials += 1
            guess_trial = guess_part[trial_index]
            dist_trial = dist_part[trial_index]
            answer = dist_trial[0]
            candidates, candidate_probs = zip(*dist_trial[1])
            max_prob = max(candidate_probs)
            if answer in candidates:
                answer_prob = candidate_probs[candidates.index(answer)]
                if answer_prob != max_prob:
                    if True in guess_trial[1]:
                        num_errors += 1
                    else:
                        print("mix-match: answer not max but True in guess_trial")
                        print("guess trial", guess_trial)
                        print("dist trial", dist_trial)
                else:
                    num_max_candidates = candidate_probs.count(max_prob)
                    if num_max_candidates != len(guess_trial[1]):
                        num_errors +=1
                    else:
                        print("mix-match: number of max candidates doesn't match")
                        print("guess trial", guess_trial)
                        print("dist trial", dist_trial)
    print(num_errors)
    print(num_trials)

def main():
    print(get_baseline_distribution("RAT", "spreading", "word"))
    print(get_baseline_distribution("WSD", "cooccurrence", "word"))

def main_integrated(agent, plot=""):
    if agent == "cooccurrence":
        print("cooccurrence, wsd, sense")
        dist = get_average_integrated_distribution("wsd", "cooccurrence", "sense", clear="never", whole_corpus=False, threshold=0,
                 var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1, func='sqrt',
                                        num_points=100)
        if plot == "wsd":
            plt.plot(dist[0], sorted(dist[1]), color="b", label="Context:Sense")

        print("cooccurrence, wsd, word")
        dist = get_average_integrated_distribution("wsd", "cooccurrence", "word", clear="never", whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        if plot == "wsd":
            plt.plot(dist[0], sorted(dist[1]), color="r", label="Context:Word")


        print("cooccurrence, rat")
        dist = get_average_integrated_distribution("rat", "cooccurrence", "word", clear="never", whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        if plot == "rat":
            plt.plot(dist[0], sorted(dist[1]), color="r")
    elif agent == "semantics":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                print("spreading, wsd, ", clear, ",", bounded)
                dist = get_average_integrated_distribution("wsd", "spreading", "word", clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
                if plot == "wsd":
                    label = "bounded: " + str(bounded) + ", clear: " + clear
                    plt.plot(dist[0], sorted(dist[1]), label=label)

        for bounded in [True, False]:
            print("spreading, rat", bounded)
            dist = get_average_integrated_distribution("rat", "spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
            if plot == "rat":
                label = "bounded: " + str(bounded)
                plt.plot(dist[0], sorted(dist[1]), label=label)
    elif agent == "cts":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                print("cts, wsd, ", clear, ",", bounded)
                get_average_integrated_distribution("wsd", "cooc_thresh_sem", "word", clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            print("cts, rat", bounded)
            get_average_integrated_distribution("rat", "cooc_thresh_sem", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    elif agent == "stc":
        for context in ["word", "sense"]:
            print("stc, wsd,", context)
            dist = get_average_integrated_distribution("wsd", "sem_thresh_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
            if plot == "wsd":
                label = "context: " + context
                plt.plot(dist[0], sorted(dist[1]), label=label)

        print("stc, rat")
        get_average_integrated_distribution("rat", "sem_thresh_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        if plot == "rat":
            plt.plot(dist[0], sorted(dist[1]))
    elif agent == "joint":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("joint, wsd, ", clear, ",", context, ",", bounded)
                    get_average_integrated_distribution("wsd", "joint_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            print("joint, rat", bounded)
            get_average_integrated_distribution("rat", "joint_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    elif agent == "add":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("add, wsd, ", clear, ",",context, ",", bounded)
                    get_average_integrated_distribution("wsd", "add_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            print("add, rat", bounded)
            get_average_integrated_distribution("rat", "add_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    elif agent == "sbc":
        for context in ["word", "sense"]:
            print("sbc, wsd,", context)
            dist = get_average_integrated_distribution("wsd", "spreading_boosted_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
            if plot == "wsd":
                label = "context: " + context
                plt.plot(dist[0], sorted(dist[1]), label=label)
        print("sbc, rat")
        dist = get_average_integrated_distribution("rat", "spreading_boosted_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        if plot == "rat":
            plt.plot(dist[0], sorted(dist[1]))
    elif agent == 'var':
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    for vartype in ["maxdiff", "stdev"]:
                        print("var, wsd, ", clear, ",", bounded, ",", context, ",", vartype)
                        get_average_integrated_distribution("wsd", "joint_var", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type=vartype, discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            for vartype in ["maxdiff", "stdev"]:
                print("var, rat", bounded, ",", vartype)
                get_average_integrated_distribution("rat", "joint_var", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    elif agent == "max":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("max, wsd, ", clear, ",",context, ",", bounded)
                    dist = get_average_integrated_distribution("wsd", "max_prob", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
                    if plot == "wsd":
                        label = "bounded: " + str(bounded) + ", clear: " + clear + ", context:", context
                        plt.plot(dist[0], sorted(dist[1]), label=label)
        for bounded in [True, False]:
            print("max, rat", bounded)
            dist = get_average_integrated_distribution("rat", "max_prob", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
            if plot == "rat":
                label = "bounded: " + str(bounded)
                plt.plot(dist[0], sorted(dist[1]), label=label)
    elif agent == "cws":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("cws, wsd, ", clear, ",",context, ",", bounded)
                    get_average_integrated_distribution("wsd", "cooc_weight_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            print("cws, rat", bounded)
            get_average_integrated_distribution("rat", "cooc_weight_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    elif agent == "ssc":
        for context in ["word", "sense"]:
            print("ssc, wsd,", context)
            dist = get_average_integrated_distribution("wsd", "spread_supplemented_cooc", context, clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
            if plot == "wsd":
                label = "context: " + context
                plt.plot(dist[0], sorted(dist[1]), label=label)
        print("ssc, rat")
        dist = get_average_integrated_distribution("rat", "spread_supplemented_cooc", "word", clear="never", whole_corpus=False,
                                        threshold=0,
                                        var_type="stdev", discount=0.1, bounded=False, num_context_acts=1,
                                        cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        if plot == "rat":
            plt.plot(dist[0], sorted(dist[1]))
    elif agent == "css":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("css, wsd, ", clear, ",",context, ",", bounded)
                    get_average_integrated_distribution("wsd", "cooc_supplemented_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            print("css, rat", bounded)
            get_average_integrated_distribution("rat", "cooc_supplemented_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    elif agent == "ces":
        for clear in ["never", "sentence", "word"]:
            for bounded in [True, False]:
                for context in ["word", "sense"]:
                    print("ces, wsd, ", clear, ",",context, ",", bounded)
                    get_average_integrated_distribution("wsd", "cooc_expanded_spreading", context, clear=clear, whole_corpus=False, threshold=0,
                                        var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1, cooc_depth=1,
                                        func='sqrt',
                                        num_points=100)
        for bounded in [True, False]:
            print("ces, rat", bounded)
            get_average_integrated_distribution("rat", "cooc_expanded_spreading", "word", clear="never", whole_corpus=False,
                                            threshold=0,
                                            var_type="stdev", discount=0.1, bounded=bounded, num_context_acts=1,
                                            cooc_depth=1,
                                            func='sqrt',
                                            num_points=100)
    plt.ylim(0, 1)
    plt.title(plot + ": " + agent)
    plt.legend(fontsize=8)
    plt.show()

for task in ["wsd", "rat"]:
    for agent in ["add", "ces", "cts", "css", "cws", "joint", "max", "sbc", "stc", "ssc", "var"]:
        print(agent, task)
        integrated_answer_comp_plot(task, agent, show=False)

# fig, ax = plt.subplots()
# deciles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#
# cooc_wsd = [0.36382919, 0.38236428, 0.40090207, 0.41945164, 0.43806716, 0.45673816,
#  0.4983495,  0.54886002, 0.62690192, 0.76148725, 0.9943867206266427]
# ax.plot(deciles, cooc_wsd, color="black", label="cooc")
#
# spreading_wsd = [0.4102787,  0.42484947, 0.43944052, 0.45444332, 0.47057282, 0.48750122,
#  0.52043501, 0.56033921, 0.61811716, 0.70925189, 0.8523690523017743]
# ax.plot(deciles, spreading_wsd, color="gray", label="sem")
#
# cfs_wsd = [0.43390563, 0.44587901, 0.45785457, 0.46988057, 0.48233072, 0.49505171,
#  0.52129156, 0.55307379, 0.60083295, 0.68106482,  0.8161292106309168]
# ax.plot(deciles, cfs_wsd, color="tab:blue", label="cfs")
#
# sfc_wsd = [0.54727424, 0.54823234, 0.54894745, 0.54945395, 0.54983816, 0.55015872,
#  0.55036475, 0.55056346, 0.55075594, 0.55094828, 0.551140610993504]
# ax.plot(deciles, sfc_wsd, color="b", label="sfc")
#
# joint_wsd = [0.36358288, 0.38209824, 0.40061482, 0.41913675, 0.4376989,  0.45629166,
#  0.49791794, 0.54841641, 0.626467,  0.76157376, 0.9966920101670949]
# ax.plot(deciles, joint_wsd, color="red", label="joint")
#
# add_wsd = [0.40988246, 0.42448535, 0.4391135,  0.45416479, 0.47034798, 0.48739383,
#  0.52029386, 0.56011906, 0.61794599, 0.70956058, 0.8538485026320248]
# ax.plot(deciles, add_wsd, color="tomato", label="add")
#
# max_wsd = [0.41034182, 0.42492418, 0.43952908, 0.45456554, 0.470767,   0.4878568,
#  0.52074284, 0.56056255, 0.61830045, 0.70919681,  0.8512246326195971]
# ax.plot(deciles, max_wsd, color="darkgoldenrod", label="max")
#
# var_wsd = [0.36366174, 0.38217665, 0.40069279, 0.41921919, 0.43778588, 0.45638436,
#  0.49801503, 0.54850452, 0.62645756, 0.76135909, 0.9963066719781765]
# ax.plot(deciles, var_wsd, color="goldenrod", label="var")
#
# css_wsd = [0.41801933, 0.43288045, 0.44775424, 0.46289147, 0.47867102, 0.49497917,
#  0.52934426, 0.57089118, 0.63247656, 0.7295846, 0.8808424710757524]
# ax.plot(deciles, css_wsd, color="purple", label="css")
#
# ssc_wsd = [0.11023338, 0.11638999, 0.12254708, 0.12870618, 0.13488332, 0.14107385,
#  0.15528224, 0.17281433, 0.20023286, 0.25369936, 0.35922443460935805]
# ax.plot(deciles, ssc_wsd, color="tab:purple", label="ssc")
#
# cws_wsd = [0.4335361,  0.44554891, 0.45756392, 0.46962868, 0.48211649, 0.49487378,
#  0.52118349, 0.55304052, 0.60094711, 0.68151112, 0.8170382572500865]
# ax.plot(deciles, cws_wsd, color="tab:green", label="cws")
#
# sbc_wsd = [0.36382919, 0.38236428, 0.40090207, 0.41945164, 0.43806706, 0.45673776,
#  0.49834869, 0.54885802, 0.62689967, 0.76148789, 0.9943899442575523]
# ax.plot(deciles, sbc_wsd, color="lime", label="sbc")
#
# ces_wsd = [0.41930535, 0.43515009, 0.45073595, 0.46593211, 0.48126791, 0.49679245,
#  0.52516067, 0.55921992, 0.60948388, 0.6916918,  0.8265744568821263]
# ax.plot(deciles, ces_wsd, color="cyan", label="ces")
#
# ax.set_title("WSD Distribution Comparisons")
# ax.legend(loc=2, fontsize="x-small")
# ax.set_xlabel("Deciles")
# plt.show()
#




#
# fig, ax = plt.subplots()
# deciles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#
# cooc_rat = [0.00869231, 0.0099596,  0.01134539, 0.01263758, 0.01411371, 0.0159939,
#  0.01987825, 0.02502264, 0.03325925, 0.05180806, 0.4509655385435877]
# ax.plot(deciles, cooc_rat, color="black", label="cooc")
#
# spreading_rat = [4.55791825e-11, 6.48474549e-08, 2.46111372e-07, 5.62333871e-07,
#  1.04397774e-06, 2.28000802e-06, 4.59820461e-06, 9.15578349e-06,
#  2.11702442e-05, 6.46206019e-05, 0.12146428431224686]
# ax.plot(deciles, spreading_rat, color="gray", label="sem")
#
# cfs_rat = [5.84060341e-12, 7.83876493e-08, 3.17846703e-07, 7.85080041e-07,
#  1.71533912e-06, 3.69604454e-06, 7.49317621e-06, 1.49425594e-05,
#  3.66934217e-05, 1.09790101e-04, 0.16222568757249525]
# ax.plot(deciles, cfs_rat, color="tab:blue", label="cfs")
#
# sfc_rat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ax.plot(deciles, sfc_rat, color="b", label="sfc")
#
# joint_rat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.33501054e-12, 0.7434701345173431]
# ax.plot(deciles, joint_rat, color="red", label="joint")
#
# add_rat = [2.23082691e-11, 2.84017554e-08, 1.09132559e-07, 2.41948778e-07,
#  4.67009349e-07, 9.99365149e-07, 2.03190006e-06, 4.12117526e-06,
#  9.65828675e-06, 2.98758112e-05, 0.23855418985199506]
# ax.plot(deciles, add_rat, color="tomato", label="add")
#
# max_rat = [2.25369852e-11, 2.86923789e-08, 1.10212119e-07, 2.44332887e-07,
#  4.70314483e-07, 1.00735507e-06, 2.05071643e-06, 4.15926574e-06,
#  9.72950259e-06, 3.00588395e-05, 0.23857633140021667]
# ax.plot(deciles, max_rat, color="darkgoldenrod", label="max")
#
# var_rat = [0.00146531, 0.00270308, 0.0040545,  0.00532144, 0.00675657, 0.00854996,
#  0.0122489,  0.01706523, 0.02455562, 0.04211039, 0.4435599557674941]
# ax.plot(deciles, var_rat, color="goldenrod", label="var")
#
# css_rat = [1.12290405e-10, 1.82732850e-07, 5.58298936e-07, 1.27981386e-06,
#  2.44355515e-06, 4.49930526e-06, 8.37670122e-06, 1.66364130e-05,
#  3.56219788e-05, 9.58564887e-05, 0.04815934669268498]
# ax.plot(deciles, css_rat, color="purple", label="css")
#
# ssc_rat = [0,0,0,0,0,0,0,0,0,0,0]
# ax.plot(deciles, ssc_rat, color="tab:purple", label="ssc")
#
# cws_rat = [8.14684762e-19, 4.11673223e-15, 3.29513011e-13, 6.64205108e-12,
#  1.01782481e-10, 1.60509376e-09, 1.63761355e-09, 2.57761775e-08,
#  1.31693051e-07, 3.46797349e-06, 0.8258110032721289]
# ax.plot(deciles, cws_rat, color="tab:green", label="cws")
#
# sbc_rat = [0.00864795, 0.00992671, 0.0113302,  0.01263686, 0.01410426, 0.01595766,
#  0.01982289, 0.02493117, 0.03244852, 0.05168483, 0.4502930975338996]
# ax.plot(deciles, sbc_rat, color="lime", label="sbc")
#
# ces_rat = [1.45553690e-10, 5.66636171e-07, 2.00336581e-06, 5.15248180e-06,
#  1.13755526e-05, 1.85657570e-05, 2.44775004e-05, 3.25601068e-05,
#  4.66638065e-05, 1.14496933e-04, 0.014038554822917102]
# ax.plot(deciles, ces_rat, color="cyan", label="ces")
#
# ax.set_title("RAT Distribution Comparisons")
# ax.legend(loc=0, fontsize="x-small")
# ax.set_xlabel("Deciles")
# ax.set_yscale("log")
# plt.show()
#
