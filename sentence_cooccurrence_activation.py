import math
from long_term_memory import ActivationDynamics
from collections import defaultdict


class SentenceCooccurrenceActivation(ActivationDynamics):
    """Activation functions to calculate the base level activation of objects with pairwise cooccurrence. Many of the
    cooccurrence functionalities are no longer used."""

    def __init__(self, ltm, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """Initialize the SentenceCooccurrenceActivation.
        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(list)
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        self.activations[mem_id].append([time, 1])
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                for element in curr_act_candidates:
                    if element is not None:
                        if element not in self.activations:
                            self.activations[element] = []
                        self.activations[element].append([time, self.activation_base ** (-graph_units)])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                else:
                                    next_act_candidates.update([list(link)[1]])
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True

    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        if mem_id not in self.activations:
            return
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = sorted(
            [[time - time_spreading_pair[0], time_spreading_pair[1]] for time_spreading_pair
             in act_times_list])
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0):
                base_act_sum_term = (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter))) + base_act_sum_term
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation


class CooccurrenceWeightedActivation(SentenceCooccurrenceActivation):
    """ Activation is the same as in the normal case, co-occurrence is just used to weight connections. """

    def __init__(self, ltm, cooc_dict, **kwargs):
        """
        Parameters:
            cooc_dict (dict): Dictionary with sorted tuple of cooccurring keys and value the conditional probability of
            those values.
        """
        super().__init__(ltm, **kwargs)
        self.cooc_dict = cooc_dict

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        self.activations[mem_id].append(
            [time, 1, 1])  # One as the 3rd element since every element co-occurs with itself 100% of the time.
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                for element in curr_act_candidates:
                    if element is not None:
                        if tuple([mem_id, element]) in self.cooc_dict:
                            cond_prob = self.cooc_dict[tuple([mem_id, element])]
                        elif tuple([element, mem_id]) in self.cooc_dict:
                            cond_prob = self.cooc_dict[tuple([mem_id, element])]
                        else:
                            cond_prob = 0
                        if element not in self.activations:
                            self.activations[element] = []
                        self.activations[element].append([time, self.activation_base ** (-graph_units), cond_prob])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                else:
                                    next_act_candidates.update([list(link)[1]])
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True

    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        if mem_id not in self.activations:
            return
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = sorted(
            [[time - time_spreading_pair[0], time_spreading_pair[1], time_spreading_pair[2]] for time_spreading_pair
             in act_times_list])
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0) and time_since_last_act_list[retrieval_pair][2] != 0:
                base_act_sum_term = time_since_last_act_list[retrieval_pair][2] * (
                (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter)))) + base_act_sum_term
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        if base_act_sum_term == 0:
            return None
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation


class BoundedActivation(SentenceCooccurrenceActivation):

    def __init__(self, ltm, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """Initialize the SentenceCooccurrenceActivation.
        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(list)
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        # Third element is the number of things that connected to this node on its activation pathway.
        self.activations[mem_id].append([time, 1, 1])
        num_memid_connections = len(list(self.ltm.knowledge[mem_id]))
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        curr_act_scaling_dict = {}
        for elem in curr_act_candidates:
            curr_act_scaling_dict[elem] = 1/num_memid_connections
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                next_act_scaling_dict = dict()
                for element in curr_act_candidates:
                    if element is not None:
                        if element not in self.activations:
                            self.activations[element] = []
                        num_elem_connections = len(list(self.ltm.knowledge[element]))
                        self.activations[element].append([time, self.activation_base ** (-graph_units), curr_act_scaling_dict[element]])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                        next_act_scaling_dict[list(item)[1]] = curr_act_scaling_dict[element] * (1/num_elem_connections)
                                else:
                                    next_act_candidates.update([list(link)[1]])
                                    next_act_scaling_dict[list(link)[1]] = curr_act_scaling_dict[element] * (
                                                1 / num_elem_connections)
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                curr_act_scaling_dict = next_act_scaling_dict
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True

    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        if mem_id not in self.activations:
            return
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = sorted(
            [[time - time_spreading_pair[0], time_spreading_pair[1], time_spreading_pair[2]] for time_spreading_pair
             in act_times_list])
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0):
                # Weighting each spreaded contribution by the inverse of the number of connections it has. Allowing only a fixed amount of
                # activation to spread throughout network
                base_act_sum_term = time_since_last_act_list[retrieval_pair][2] * (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter))) + base_act_sum_term
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation


class CooccurrenceWeightedBoundedActivation(SentenceCooccurrenceActivation):

    def __init__(self, ltm, cooc_dict, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """
        Parameters:
            cooc_dict (dict): Dictionary with sorted tuple of cooccurring keys and value the conditional probability of
            those values.
        """
        super().__init__(ltm, **kwargs)
        self.activations = defaultdict(list)
        self.cooc_dict = cooc_dict
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        self.activations[mem_id].append([time, 1, 1, 1])
                            # One as the 3rd element since every element co-occurs with itself 100% of the time.
                            # 4th element is the number of things that connected to this node on its activation pathway.
        num_memid_connections = len(list(self.ltm.knowledge[mem_id]))
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        curr_act_scaling_dict = {}
        for elem in curr_act_candidates:
            curr_act_scaling_dict[elem] = 1 / num_memid_connections
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                next_act_scaling_dict = dict()
                for element in curr_act_candidates:
                    if element is not None:
                        if type(mem_id) == str:
                            mem_id = mem_id.upper()
                        if type(element) == str:
                            element = element.upper()
                        if tuple([mem_id, element]) in self.cooc_dict:
                            cond_prob = self.cooc_dict[tuple([mem_id, element])]
                        elif tuple([element, mem_id]) in self.cooc_dict:
                            cond_prob = self.cooc_dict[tuple([element, mem_id])]
                        else:
                            cond_prob = 0
                        if element not in self.activations:
                            self.activations[element] = []
                        num_elem_connections = len(list(self.ltm.knowledge[element]))
                        self.activations[element].append([time, self.activation_base ** (-graph_units), cond_prob,
                                                          curr_act_scaling_dict[element]])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                        next_act_scaling_dict[list(item)[1]] = curr_act_scaling_dict[element] * (
                                                    1 / num_elem_connections)
                                else:
                                    next_act_candidates.update([list(link)[1]])
                                    next_act_scaling_dict[list(link)[1]] = curr_act_scaling_dict[element] * (
                                            1 / num_elem_connections)
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                curr_act_scaling_dict = next_act_scaling_dict
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True


    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        if mem_id not in self.activations:
            return
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = sorted(
            [[time - time_spreading_pair[0], time_spreading_pair[1], time_spreading_pair[2], time_spreading_pair[3]] for
             time_spreading_pair in act_times_list])
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0) and time_since_last_act_list[retrieval_pair][2] != 0:
                base_act_sum_term = (time_since_last_act_list[retrieval_pair][3] *
                                     time_since_last_act_list[retrieval_pair][2] * (
                                     (time_since_last_act_list[retrieval_pair][1] * (
                                     time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter)))) + base_act_sum_term)
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        if base_act_sum_term == 0:
            return None
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation


class CoocSupplementedActivation(SentenceCooccurrenceActivation):
    def __init__(self, ltm, cooc_dict, discount, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """Initialize the SentenceCooccurrenceActivation.
        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(ltm, **kwargs)
        self.cooc_dict = cooc_dict
        self.discount = discount
        self.activations = defaultdict(list)
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        self.activations[mem_id].append([time, 1, 1])
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                for element in curr_act_candidates:
                    if element is not None:
                        if element not in self.activations:
                            self.activations[element] = []
                        self.activations[element].append([time, self.activation_base ** (-graph_units), 1])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                else:
                                    next_act_candidates.update([list(link)[1]])
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True

    def activate_cooc_word(self, mem_id, spread_depth=-1, time=0):
        if self.ltm.knowledge.get(mem_id) is None:
            return False
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        self.activations[mem_id].append([time, 1, self.discount])
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                for element in curr_act_candidates:
                    if element is not None:
                        if element not in self.activations:
                            self.activations[element] = []
                        self.activations[element].append([time, self.activation_base ** (-graph_units), self.discount])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                else:
                                    next_act_candidates.update([list(link)[1]])
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True



    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        if mem_id not in self.activations:
            return
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = sorted(
            [[time - time_spreading_pair[0], time_spreading_pair[1], time_spreading_pair[2]] for time_spreading_pair
             in act_times_list])
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0):
                base_act_sum_term = time_since_last_act_list[retrieval_pair][2] * (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter))) + base_act_sum_term
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation


class CoocBoundedSupplementedActivation(SentenceCooccurrenceActivation):
    def __init__(self, ltm, cooc_dict, discount, constant_offset=0, activation_base=2, decay_parameter=0.05, **kwargs):
        """Initialize the SentenceCooccurrenceActivation.
        Parameters:
            ltm (LongTermMemory): The LongTermMemory that will be using this activation.
            activation_base (float): A parameter in the activation equation.
            decay_parameter (float): A parameter in the activation equation.
            constant_offset (float): A parameter in the activation equation.
        """
        super().__init__(ltm, **kwargs)
        self.cooc_dict = cooc_dict
        self.discount = discount
        self.activations = defaultdict(list)
        self.constant_offset = constant_offset
        self.activation_base = activation_base
        self.decay_parameter = decay_parameter

    def simple_activate(self, mem_id, spread_depth=-1, time=0):
        """
        Activates a given element and its neighbors via spreading activation.
        Parameters:
            mem_id (any): The ID of the desired element.
            spread_depth (int): The depth of connections to activate when a given element is activated. Serves mainly to
                allow the same setup for non-spreading and spreading scenarios.
            time (float): The time of retrieval (for activation) (optional)
        Returns:
            True: If completed.
        """
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        # Third element corresponds to decay for cooc related words (not in this function),
        # Fourth element corresponds to weight of bounded activation
        self.activations[mem_id].append([time, 1, 1, 1])
        num_memid_connections = len(list(self.ltm.knowledge[mem_id]))
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        curr_act_scaling_dict = {}
        for elem in curr_act_candidates:
            curr_act_scaling_dict[elem] = 1 / num_memid_connections
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                next_act_scaling_dict = dict()
                for element in curr_act_candidates:
                    if element is not None:
                        if element not in self.activations:
                            self.activations[element] = []
                        num_elem_connections = len(list(self.ltm.knowledge[element]))
                        self.activations[element].append([time, self.activation_base ** (-graph_units), 1, curr_act_scaling_dict[element]])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                        next_act_scaling_dict[list(item)[1]] = curr_act_scaling_dict[element] * (
                                                    1 / num_elem_connections)
                                else:
                                    next_act_candidates.update([list(link)[1]])
                                    next_act_scaling_dict[list(link)[1]] = curr_act_scaling_dict[element] * (
                                            1 / num_elem_connections)
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                curr_act_scaling_dict = next_act_scaling_dict
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True

    def activate_cooc_word(self, mem_id, spread_depth=-1, time=0):
        if self.ltm.knowledge.get(mem_id) is None:
            return False
        if mem_id not in self.activations:
            self.activations[mem_id] = []
        # Third element corresponds to cooc discount... fourth element corresponds to proportion of spreading each element
        # gets
        self.activations[mem_id].append([time, 1, self.discount, 1])
        num_memid_connections = len(list(self.ltm.knowledge[mem_id]))
        prev_act_candidates = set([mem_id])  # Candidates that have been activated before (prevents infinite looping)
        # Candidates to activate next (for spreading)
        curr_act_candidates = [list(element)[1] for element in list(self.ltm.knowledge.get(mem_id))]
        curr_act_scaling_dict = {}
        for elem in curr_act_candidates:
            curr_act_scaling_dict[elem] = 1 / num_memid_connections
        if spread_depth != 0:  # if spreading is allowed...
            graph_units = 1  # distance from originally activated node
            while curr_act_candidates:  # checking that spreading still needs to be done
                next_act_candidates = set()
                next_act_scaling_dict = dict()
                for element in curr_act_candidates:
                    if element is not None:
                        if element not in self.activations:
                            self.activations[element] = []
                        num_elem_connections = len(list(self.ltm.knowledge[element]))
                        self.activations[element].append([time, self.activation_base ** (-graph_units), self.discount,
                                                          curr_act_scaling_dict[element]])
                        # Add to next things to activate, the connections of the element we just activated
                        new_links = self.ltm.knowledge.get(element)
                        if new_links is not None:
                            for link in list(new_links):
                                if type(link) == list:
                                    for item in link:
                                        next_act_candidates.update([list(item)[1]])
                                        next_act_scaling_dict[list(item)[1]] = curr_act_scaling_dict[element] * (
                                                    1 / num_elem_connections)
                                else:
                                    next_act_candidates.update([list(link)[1]])
                                    next_act_scaling_dict[list(link)[1]] = curr_act_scaling_dict[element] * (
                                            1 / num_elem_connections)
                # If we don't want to spread farther - stop going through connections and activating them
                if graph_units == spread_depth:
                    break
                graph_units += 1  # Moving to the next "round" of connections
                prev_act_candidates.update(curr_act_candidates)
                # Making sure that we haven't already activated the elements we are to "spread" to next...
                curr_act_candidates = next_act_candidates.difference(prev_act_candidates)
                curr_act_scaling_dict = next_act_scaling_dict
                # If there's no more elements to activate, done!
                if curr_act_candidates == [] or curr_act_candidates is None:
                    break
        return True



    def get_activation(self, mem_id, time):
        """Get the activation of the element with the given ID.
        Parameters:
            mem_id (any): The ID of the desired element.
            time (int): The time of activation. Optional.
        Returns:
            float: The activation of the element.
        """
        if mem_id not in self.activations:
            return
        act_times_list = self.activations[mem_id]
        if act_times_list == [] or (len(act_times_list) == 1 and act_times_list[0][0] == 0):
            return None
        # We first create a nested list where each entry is a list: [time since last activation,
        # "graph distance multiplier"] where the "graph distance multiplier" is a previously calculated indicator of how
        # far away from the originally activated word the word being activated is.
        time_since_last_act_list = sorted(
            [[time - params[0], params[1], params[2], params[3]] for params
             in act_times_list])
        base_act_sum_term = 0
        # For every activation...
        for retrieval_pair in range(len(time_since_last_act_list)):
            # Using the base activation equation to calculate the term referring to each activation
            if (act_times_list[retrieval_pair][0] > 0):
                base_act_sum_term = (time_since_last_act_list[retrieval_pair][3] *
                                     time_since_last_act_list[retrieval_pair][2] *
                                     (time_since_last_act_list[retrieval_pair][1] * (
                        time_since_last_act_list[retrieval_pair][0] ** (-self.decay_parameter))) + base_act_sum_term)
        # To finish calculating the activation of the element in question, adding the constant offset (normally 0) and
        # then taking the log of the aggregate term calculated in the for loop above.
        base_level_activation = self.constant_offset + math.log(base_act_sum_term)
        return base_level_activation