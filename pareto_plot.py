from integrated_mechanisms_RAT import *
from integrated_mechanisms_WSD import *
import matplotlib.pyplot as plt
from n_gram_cooccurrence.google_ngrams import *
import json
import os
import matplotlib.ticker as mtick
import numpy as np


def compute_discounted_accuracy_wsd(guess_method, context_type='sense', spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0, whole_corpus=False, threshold=0,
                 var_type="stdev", discount=0.1, activate_answer=False, activate_sentence_words=True, bounded=False,
                 num_context_acts=1, cooc_depth=1, func='sqrt'):
    # files should be a list of 6 string filenames...
    files = get_wsd_filenames(guess_method, context_type, spreading, clear, activation_base, decay_parameter,
                              constant_offset, whole_corpus, threshold, var_type, discount, activate_answer,
                              activate_sentence_words, bounded, num_context_acts, cooc_depth, func)
    cumulative_acc = 0
    num_words = 0
    for file in files:
        total_guesses = 0
        discounted_count = 0
        guess_list = json.load(open(file))
        for word_guess in guess_list:
            guesses = word_guess[1]
            for guess in guesses:
                total_guesses += 1
            if any(guesses):
                discounted_count += 1 / len(guesses)
        partition_words = len(guess_list)
        num_words += partition_words
        cumulative_acc += (discounted_count / total_guesses) * partition_words
    print(cumulative_acc/num_words, guess_method, "context", context_type, "clear", clear, "bounded", bounded, "var type", var_type)
    return cumulative_acc / num_words

def get_wsd_filenames(guess_method, context_type='sense', spreading=True, clear="never",
                 activation_base=2, decay_parameter=0.05, constant_offset=0, whole_corpus=False, threshold=0,
                 var_type="stdev", discount=0.1, activate_answer=False, activate_sentence_words=True, bounded=False,
                 num_context_acts=1, cooc_depth=1, func='sqrt'):
    guess_list_files = []
    for partition in [1,2,3,4,5,6]:
        wsd = WSDTask(5000, partition)
        agent = create_WSD_agent(guess_method=guess_method, partition=partition, num_sentences=5000, context_type=context_type,
                                                       spreading=spreading, clear=clear, activation_base=activation_base, decay_parameter=decay_parameter,
                                                       constant_offset=constant_offset, whole_corpus=whole_corpus, threshold=threshold, var_type=var_type, discount=discount,
                                                       activate_answer=activate_answer, activate_sentence_words=activate_sentence_words, bounded=bounded, num_context_acts=num_context_acts,
                                                       cooc_depth=cooc_depth, func=func)
        filename = 'results/' +  wsd.to_string_id() + "_" + agent.to_string_id() + ".json"
        if not os.path.exists(filename):
            task = WSDTask(5000, partition)
            task.run(agent)
        guess_list_files.append(filename)
    return guess_list_files

def compute_discounted_accuracy_rat(guess_method, source='SFFAN', spreading=True, activation_base=2, decay_parameter=0.05,
                 constant_offset=0, ngrams=GoogleNGram('~/ngram'), threshold=0, var_type='stdev', discount=0.1,
                                    bounded=False, cooc_depth=1, func='sqrt'):
    filename = get_rat_filename(guess_method, source, spreading, activation_base, decay_parameter,
                 constant_offset, ngrams, threshold, var_type, discount, bounded, cooc_depth, func)
    guess_list = json.load(open(filename))
    discounted_count = 0
    for word_guess in guess_list:
        guesses = word_guess[1]
        if any(guesses):
            discounted_count += 1 / len(guesses)
    print(discounted_count/len(guess_list), guess_method, "var type", var_type)
    return discounted_count / len(guess_list)

def get_rat_filename(guess_method, source='SFFAN', spreading=True, activation_base=2, decay_parameter=0.05,
                 constant_offset=0, ngrams=GoogleNGram('~/ngram'), threshold=0, var_type='stdev', discount=0.1,
                     bounded=False, cooc_depth=1, func='sqrt'):
    agent = create_RAT_agent(guess_method=guess_method, source=source, spreading=spreading, activation_base=activation_base, decay_parameter=decay_parameter,
                 constant_offset=constant_offset, ngrams=ngrams, threshold=threshold, var_type=var_type, discount=discount, bounded=bounded, cooc_depth=cooc_depth, func=func)
    filename = './results/' + 'RAT_' + agent.to_string_id() + ".json"
    if not os.path.exists(filename):
        task = RatTest()
        task.run(agent)
    return filename

def get_results(guess_method, wsd_context=[], wsd_clear=[], vartype="", bounded=False, spreading=True):
    WSD_filename= "./accuracies/WSD_" + guess_method + "_" + vartype + "_" + str(bounded) + "_" + str(spreading) + "_" + str(len(wsd_clear)) + "_" + str(len(wsd_context)) + ".json"
    if not os.path.isfile(WSD_filename):
        wsd_accs = []
        if not wsd_context and not wsd_clear:
            wsd_accs.append(
                compute_discounted_accuracy_wsd(guess_method, context_type="", spreading=spreading, clear="",
                                                activation_base=2, decay_parameter=0.05, constant_offset=0,
                                                whole_corpus=False, threshold=0,
                                                var_type=vartype, discount=0.1, activate_answer=False,
                                                activate_sentence_words=True, bounded=bounded))
            print("wsd", guess_method, wsd_accs[-1])
        elif not wsd_context:
            for clear in wsd_clear:
                wsd_accs.append(
                        compute_discounted_accuracy_wsd(guess_method, context_type="", spreading=spreading, clear=clear,
                                                    activation_base=2, decay_parameter=0.05, constant_offset=0,
                                                    whole_corpus=False, threshold=0,
                                                    var_type=vartype, discount=0.1, activate_answer=False,
                                                    activate_sentence_words=True, bounded=bounded))
                print("wsd", guess_method, clear, bounded, wsd_accs[-1])
        elif not wsd_clear:
            for context in wsd_context:
                wsd_accs.append(
                    compute_discounted_accuracy_wsd(guess_method, context_type=context, spreading=spreading, clear="",
                                                    activation_base=2, decay_parameter=0.05, constant_offset=0,
                                                    whole_corpus=False, threshold=0,
                                                    var_type=vartype, discount=0.1, activate_answer=False,
                                                    activate_sentence_words=True, bounded=bounded))
                print("wsd", guess_method, bounded, context, wsd_accs[-1])
        else:
            for clear in wsd_clear:
                for context in wsd_context:
                    wsd_accs.append(
                        compute_discounted_accuracy_wsd(guess_method, context_type=context, spreading=spreading, clear=clear,
                                                    activation_base=2, decay_parameter=0.05, constant_offset=0,
                                                    whole_corpus=False, threshold=0,
                                                    var_type=vartype, discount=0.1, activate_answer=False,
                                                    activate_sentence_words=True, bounded=bounded))
                    print("wsd", guess_method, clear, bounded, context, wsd_accs[-1])
        print("accs", wsd_accs)
        json.dump(wsd_accs, open(WSD_filename, "w"))
    else:
        wsd_accs = json.load(open(WSD_filename))
    RAT_filename = "./accuracies/RAT_" + guess_method + "_" + vartype + "_" + str(bounded) + "_" + str(spreading) + "_" + str(len(wsd_clear)) + "_" + str(len(wsd_context)) + ".json"
    if not os.path.isfile(RAT_filename):
        rat_accs = [compute_discounted_accuracy_rat(guess_method, source='SFFAN', spreading=spreading, activation_base=2,
                                                decay_parameter=0.05,
                                                constant_offset=0, ngrams=GoogleNGram('~/ngram'), threshold=0,
                                                var_type=vartype, discount=0.1, bounded=bounded)]
        print("rat", guess_method, bounded, rat_accs)
        rat_accs = rat_accs * len(wsd_accs)
        print("rat accs", rat_accs)
        json.dump(rat_accs, open(RAT_filename, "w"))
    else:
        rat_accs = json.load(open(RAT_filename))
    return wsd_accs, rat_accs

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

# # Make plot...
# fig, ax = plt.subplots()
# ax.set_title("RAT vs. WSD Accuracy for Retrieval Mechanisms")
# ax.set_xlim([0, 1.02])
# ax.set_ylim([-0.01, 0.45])
# ax.set_ylabel("RAT Accuracy")
# ax.set_xlabel("WSD Accuracy")
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
# ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
#
# data = {}
#
# wsd_results_lb = 0.3794224297
# rat_results_lb = 0.030967917088331152
# data["random uniform"] = [[wsd_results_lb], [rat_results_lb]]
#
#
# print("oracle")
# wsd, rat = get_results(guess_method="oracle", wsd_context=[], wsd_clear=[], vartype="", bounded=False, spreading=True)
# wsd, rat = pareto_filter(wsd, rat)
# print("oracle", wsd, rat)
# ax.scatter(wsd, rat, marker='^', color="black", s=80, label="Oracle")
# ax.scatter(wsd_results_lb, rat_results_lb, marker='^', color="black", s=80, label="Uniform Random")
# data["oracle"] = [wsd, rat]
#
#
# print("cooccurrence")
# wsd, rat = get_results(guess_method="cooccurrence", wsd_context=["word", "sense"], wsd_clear=[], vartype="", bounded=False, spreading=True)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="black", label="Co-occurrence")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# print("wsd", wsd)
# print("rat", rat)
# ax.plot(wsd, rat, color="black")
# data["cooccurrence"] = [wsd, rat]
#
# print("spreading")
# wsd, rat = get_results(guess_method="spreading", wsd_context=[], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="spreading", wsd_context=[], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="black", label="Semantic Spreading")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="black")
# data["semantic"] = [wsd, rat]
#
# print("cts")
# wsd, rat = get_results(guess_method="cooc_thresh_sem", wsd_context=[], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="cooc_thresh_sem", wsd_context=[], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="tab:blue", label="CTS")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="tab:blue")
# data["cts"] = [wsd, rat]
#
# print("stc")
# wsd, rat = get_results(guess_method="sem_thresh_cooc", wsd_context=["word", "sense"], wsd_clear=[], vartype="", bounded=False, spreading=True)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="b", label="STC")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="b")
# data["stc"] = [wsd, rat]
#
# print("joint")
# wsd, rat = get_results(guess_method="joint_prob", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="joint_prob", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="red", label="Joint")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="red")
# data["jpr"] = [wsd, rat]
#
# print("add")
# wsd, rat = get_results(guess_method="add_prob", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="add_prob", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="tomato", label="Additive")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="tomato")
# data["apr"] = [wsd, rat]
#
# print("spreading boosted cooc")
# wsd, rat = get_results(guess_method="spreading_boosted_cooc", wsd_context=["word", "sense"], wsd_clear=[], vartype="", bounded=False, spreading=True)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="lime", label="SBC")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# print("wsd", wsd)
# print("rat", rat)
# ax.plot(wsd, rat, color="lime")
# data["sbc"] = [wsd, rat]
#
# print("variance")
# wsd, rat = get_results(guess_method="joint_var", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="maxdiff", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="joint_var", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="maxdiff", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd3, rat3 = get_results(guess_method="joint_var", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="stdev", bounded=True, spreading=True)
# wsd.extend(wsd3)
# rat.extend(rat3)
# wsd4, rat4 = get_results(guess_method="joint_var", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="stdev", bounded=False, spreading=True)
# wsd.extend(wsd4)
# rat.extend(rat4)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="goldenrod", label="Variance")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# print("wsd", wsd)
# print("rat", rat)
# ax.plot(wsd, rat, color="goldenrod")
# data["vbs"] = [wsd, rat]
#
# print("max")
# wsd, rat = get_results(guess_method="max_prob", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="max_prob", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="darkgoldenrod", label="Max Probability")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="darkgoldenrod")
# data["mpr"] = [wsd, rat]
#
# print("cooc weight spreading")
# wsd, rat = get_results(guess_method="cooc_weight_spreading", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="cooc_weight_spreading", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="tab:green", label="CWS")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="tab:green")
# data["cws"] = [wsd, rat]
#
# print("spread supp cooc")
# wsd, rat = get_results(guess_method="spread_supplemented_cooc", wsd_context=["word", "sense"], wsd_clear=[], vartype="", bounded=False, spreading=True)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="tab:purple", label="SSC")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="tab:purple")
# data["ssc"] = [wsd, rat]
#
# print("cooc supp spread")
# wsd, rat = get_results(guess_method="cooc_supplemented_spreading", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="cooc_supplemented_spreading", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# ax.scatter(wsd, rat, marker='o', color="purple", label="CSS")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="purple")
# data["css"] = [wsd, rat]
#
# print("cooc expanded spreading")
# wsd, rat = get_results(guess_method="cooc_expanded_spreading", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=True, spreading=True)
# wsd2, rat2 = get_results(guess_method="cooc_expanded_spreading", wsd_context=["word", "sense"], wsd_clear=["word", "sentence", "never"], vartype="", bounded=False, spreading=True)
# wsd.extend(wsd2)
# rat.extend(rat2)
# wsd, rat = pareto_filter(wsd, rat)
# print("wsd", wsd)
# print("rat", rat)
# ax.scatter(wsd, rat, marker='o', color="cyan", label="CES")
# #wsd, rat = zip(*sorted(zip(wsd, rat)))
# ax.plot(wsd, rat, color="cyan")
# data["ces"] = [wsd, rat]
#
#
# with open("/Users/lilygebhart/Downloads/wsd_rat_data.json", 'w') as f:
#     json.dump(data, f)
#
# plt.subplots_adjust(right=0.8)
# #ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", prop={'size': 6})
# plt.show()

