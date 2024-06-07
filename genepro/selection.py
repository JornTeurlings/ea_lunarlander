import itertools

import Levenshtein
from multiprocessing import Pool
import numpy as np
from copy import deepcopy
import random
import math

from genepro.multitree import Multitree


def elitist_selection(contestants: list, num_to_select: int) -> list:
    """
    Performs elitism selection by selecting the top `num_to_select` individuals based on fitness.

    Parameters
    ----------
    contestants : list
        list of Node containing trees that undergo the selection
    num_to_select : int
        how many should be selected

    Returns
    -------
    list
        list containing the top `num_to_select` trees based on fitness
    """
    sorted_contestants = sorted(contestants, key=lambda x: x.fitness, reverse=True)

    selected = sorted_contestants[:num_to_select]

    return selected


def exponential_ranking_selection_with_elitism(contestants, num_to_select, c, k, elitism_rate):
    # Calculate number of elites to keep
    num_elites = int(len(contestants) * elitism_rate)
    elites = elitist_selection(contestants, num_elites)

    # The rest of the population to participate in exponential ranking selection
    non_elites = [c for c in contestants if c not in elites]
    selected = list()

    # Ensure we have enough contestants for selection
    assert len(non_elites) >= num_to_select - num_elites

    # Compute selection probabilities
    n = len(non_elites)
    probs = [0.0] * n
    non_elites = sorted(non_elites, key=lambda x: x.fitness)
    for i in range(len(non_elites)):
        probs[i] = (c ** (n - i + 1)) * (c - 1) / (c ** n - 1)

    # Select individuals using probabilities
    chosen = random.choices(non_elites, probs, k=num_to_select - num_elites)
    selected.append(deepcopy(chosen))
    non_elites.remove(chosen)
    # Add elites directly to the selected pool
    selected.extend(elites)

    return selected


def linear_ranking_selection_with_elitism(contestants, num_to_select, s, elitism_rate):
    # Calculate number of elites to keep
    num_elites = int(len(contestants) * elitism_rate)
    elites = elitist_selection(contestants, num_elites)

    # The rest of the population to participate in linear ranking selection
    non_elites = [c for c in contestants if c not in elites]
    selected = list()

    # Ensure we have enough contestants for selection
    assert len(non_elites) >= num_to_select - num_elites

    # Compute selection probabilities
    n = len(non_elites)
    probs = [0.0] * n
    non_elites = sorted(non_elites, key=lambda x: x.fitness)
    for i in range(len(non_elites)):
        probs[i] = (1 / n) * (s - ((2 * s - 2) * i) / (n - 1))

    # Select individuals using probabilities
    chosen = random.choices(non_elites, probs, k=num_to_select - num_elites)
    selected.append(deepcopy(chosen))
    non_elites.remove(chosen)

    # Add elites directly to the selected pool
    selected.extend(elites)

    return selected


def roulette_wheel_selection(contestants, num_to_select, with_stochastic_acc):
    """
    Performs random selection of contestants by doing proportionate selection according to their fitness.

    Parameters
    ----------
      contestants : list
        list of Node containing trees that undergo the selection
      num_to_select : int
        how many should be selected
      with_stochastic_acc : bool
        whether stochastic acceptance should be used

    Returns
    -------
    list
        list containing random sampled trees that were used by the wheel
    """
    selected = []
    n = len(contestants)
    q = 0
    fitness_list = [contestant.fitness for contestant in contestants]

    # Ensure all fitness values are positive
    min_fitness = min(fitness_list)
    shift_value = abs(min_fitness) + 1
    shifted_fitness = [f + shift_value for f in fitness_list]

    total_fitness = np.sum(shifted_fitness)
    maximum_fitness = np.max(shifted_fitness)
    mean_fitness = np.mean(shifted_fitness)

    # Rejection probability for Stochastic Acceptance
    if with_stochastic_acc:
        q = 1 - (mean_fitness / maximum_fitness)

    wheel = []
    sum_probs = 0.0

    for i in range(len(contestants)):
        if with_stochastic_acc:
            prob = shifted_fitness[i] / (n * maximum_fitness * (1 - q))
        else:
            prob = shifted_fitness[i] / total_fitness
        sum_probs += prob
        wheel.append(sum_probs)

    for _ in range(num_to_select):
        chosen = 0
        prob = np.random.rand()
        for j in range(len(wheel)):
            if prob <= wheel[j]:
                chosen = j
                break
        selected.append(contestants[chosen])

    return selected


def random_selection(contestants: list, num_to_select: int) -> list:
    """
    Performs random selection of contestants by uniformly sampling from the population.

    Parameters
    ----------
      contestants : list
        list of Node containing trees that undergo the selection
      num_survivors : int
        how many should be selected

    Returns
    --------
    list
        list containing random sampled trees that were selected
    """
    selected = list()
    n = len(contestants)
    assert n >= num_to_select
    for i in range(num_to_select):
        selected.append(contestants[np.random.randint(0, n)])

    return selected


def tournament_selection_with_elitism(contestants, num_to_select, tournament_size=4, elitism_rate=0.05):
    # Calculate number of elites to keep
    num_elites = int(len(contestants) * elitism_rate)
    elites = elitist_selection(contestants, num_elites)

    # The rest of the population to participate in tournament selection
    non_elites = [c for c in contestants if c not in elites]
    selected = list()

    # Ensure we have enough contestants for the tournament
    assert len(non_elites) >= num_to_select - num_elites

    # Select from non-elites to fill the remaining slots
    num_remaining_select = num_to_select - num_elites
    n = len(non_elites)
    num_selected_per_parse = n // tournament_size
    num_parses = num_remaining_select // num_selected_per_parse

    assert n / tournament_size == num_selected_per_parse, "Number of non-elites {} is not a multiple of tournament size {}".format(
        n, tournament_size)
    assert num_remaining_select / num_selected_per_parse == num_parses

    for _ in range(num_parses):
        # shuffle
        np.random.shuffle(non_elites)
        fitnesses = np.array([t.fitness for t in non_elites])

        winning_indices = np.argmax(fitnesses.reshape((-1, tournament_size)), axis=1)
        winning_indices += np.arange(0, n, tournament_size)

        selected += [deepcopy(non_elites[i]) for i in winning_indices]

    # Add elites directly to the selected pool
    selected.extend(elites)

    return selected


distance_cache = {}


def compute_edit_distance(seq1: str, seq2: str) -> int:
    cache_key = (seq1, seq2)
    if cache_key in distance_cache:
        return distance_cache[cache_key]
    distance = Levenshtein.distance(seq1, seq2)
    distance_cache[cache_key] = distance
    return distance


def compute_average_pairwise_distance(tree1: Multitree, tree2: Multitree) -> int:
    seq1 = tree1.get_readable_repr()
    seq2 = tree2.get_readable_repr()
    n = len(seq2)

    return np.mean([compute_edit_distance(seq1[i], seq2[i]) for i in range(n)])


def pairwise_distance(pair):
    tree1, tree2 = pair
    return compute_average_pairwise_distance(tree1, tree2)


def compute_all_pairwise_distance(population):
    with Pool() as pool:
        pairs = list(itertools.combinations(population, 2))
        distances = pool.map(pairwise_distance, pairs)

    average_distance = np.mean(distances)
    return average_distance
