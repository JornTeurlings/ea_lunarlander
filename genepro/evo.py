from enum import Enum
from typing import Callable

import numpy as np
from numpy.random import random as randu
from numpy.random import randint as randi
from numpy.random import choice as randc
from numpy.random import shuffle
import time, inspect
from copy import deepcopy
from joblib.parallel import Parallel, delayed

import torch
import torch.optim as optim

from genepro.node import Node
from genepro.variation import *
from genepro.selection import tournament_selection_with_elitism, compute_all_pairwise_distance

from collections import namedtuple
import copy

class OptimizationBehavior(Enum):
  NONE = 0
  ALL_INDIVIDUALS = 1
  SAMPLED_INDIVIDUALS = 2
  ALL_PARENTS = 3
  SAMPLED_PARENTS = 4
  ALL_OFFSPRING = 5
  SAMPLED_OFFSPRING = 6

class Evolution:
  """
  Class concerning the overall evolution process.

  Parameters
  ----------
  fitness_function : function
    the function used to evaluate the quality of evolving trees, should take a Node and return a float; higher fitness is better

  internal_nodes : list
    list of Node objects to be used as internal nodes for the trees (e.g., [Plus(), Minus(), ...])

  leaf_nodes : list
    list of Node objects to be used as leaf nodes for the trees (e.g., [Feature(0), Feature(1), Constant(), ...])

  pop_size : int, optional
    the population size (default is 256)

  init_max_depth : int, optional
    the maximal depth trees can have at initialization (default is 4)

  max_tree_size : int, optional
    the maximal number of nodes trees can have during the entire evolution (default is 64)

  crossovers : list, optional
    list of dictionaries that contain: "fun": crossover functions to be called, "rate": rate of applying crossover, "kwargs" (optional): kwargs for the chosen crossover function (default is [{"fun":subtree_crossover, "rate": 0.75}])

  mutations : list, optional
    similar to `crossovers`, but for mutation (default is [{"fun":subtree_mutation, "rate": 0.75}])

  coeff_opts : list, optional
    similar to `crossovers`, but for coefficient optimization (default is [{"fun":coeff_mutation, "rate": 1.0}])

  selection : dict, optional
    dictionary that contains: "fun": function to be used to select promising parents, "kwargs": kwargs for the chosen selection function (default is {"fun":tournament_selection,"kwargs":{"tournament_size":4}})

  max_evals : int, optional
    termination criterion based on a maximum number of fitness function evaluations being reached (default is None)

  max_gens : int, optional
    termination criterion based on a maximum number of generations being reached (default is 100)

  max_time: int, optional
    termination criterion based on a maximum runtime being reached (default is None)

  n_jobs : int, optional
    number of jobs to use for parallelism (default is 4)

  verbose : bool, optional
    whether to log information during the evolution (default is False)

  Attributes
  ----------
  All of the parameters, plus the following:

  population : list
    list of Node objects that are the root of the trees being evolved

  num_gens : int
    number of generations

  num_evals : int
    number of evaluations

  start_time : time
    start time

  elapsed_time : time
    elapsed time

  best_of_gens : list
    list containing the best-found tree in each generation; note that the entry at index 0 is the best at initialization
  """
  def __init__(self,
    # required settings
    fitness_function : Callable[[Node], float],
    internal_nodes : list,
    leaf_nodes : list,
    # optional evolution settings
    n_trees : int,
    pop_size : int=256,
    init_max_depth : int=4,
    max_tree_size : int=64,
    crossovers : list=[{"fun":subtree_crossover, "rate": 0.5}],
    mutations : list= [{"fun":subtree_mutation, "rate": 0.5}],
    coeff_opts : list = [{"fun":coeff_mutation, "rate": 0.5}],
    selection : dict={"fun":tournament_selection_with_elitism,"kwargs":{"tournament_size":8}},
    # termination criteria
    max_evals : int=None,
    max_gens : int=100,
    max_time : int=None,
    # optimization settings
    optimization_behavior : OptimizationBehavior=OptimizationBehavior.NONE,
    batch_size : int=64,
    GAMMA : float=0.99,
    steps : int=10,
    lr : float=0.05,
    # other
    n_jobs : int=4,
    verbose : bool=False,
    ):

    # set parameters as attributes
    _, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop('self')
    for arg, val in values.items():
      setattr(self, arg, val)

    # fill-in empty kwargs if absent in crossovers, mutations, coeff_opts
    for variation_list in [crossovers, mutations, coeff_opts]:
      for i in range(len(variation_list)):
        if "kwargs" not in variation_list[i]:
          variation_list[i]["kwargs"] = dict()
    # same for selection
    if "kwargs" not in selection:
      selection["kwargs"] = dict()

    # initialize some state variables
    self.population = list()
    self.num_gens = 0
    self.num_evals = 0
    self.start_time, self.elapsed_time = 0, 0
    self.best_of_gens = list()
    self.best_fitness = list()
    self.average_of_gens = list()
    self.diversity_of_fitness = list()
    self.diversity_of_population = list()

    self.memory = None


  def _must_terminate(self) -> bool:
    """
    Determines whether a termination criterion has been reached

    Returns
    -------
    bool
      True if a termination criterion is met, else False
    """
    self.elapsed_time = time.time() - self.start_time
    if self.max_time and self.elapsed_time >= self.max_time:
      return True
    elif self.max_evals and self.num_evals >= self.max_evals:
      return True
    elif self.max_gens and self.num_gens >= self.max_gens:
      return True
    return False

  def _initialize_population(self):
    """
    Generates a random initial population and evaluates it
    """
    # initialize the population
    self.population = Parallel(n_jobs=self.n_jobs)(
        delayed(generate_random_multitree)(self.n_trees,
          self.internal_nodes, self.leaf_nodes, max_depth=self.init_max_depth )
        for _ in range(self.pop_size))

    for count, individual in enumerate(self.population):
      individual.get_readable_repr()

    # evaluate the trees and store their fitness
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in self.population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      self.population[i].fitness = fitnesses[i]
      # store eval cost
    self.num_evals += self.pop_size
    # store best at initialization
    fitness_list = [t.fitness for t in self.population]
    best = self.population[np.argmax(fitness_list)]
    self.best_of_gens.append(deepcopy(best))
    average = np.mean(fitness_list)
    variation = np.std(fitness_list)
    diversity_gen = compute_all_pairwise_distance(self.population)

    self.diversity_of_population.append(diversity_gen)
    self.best_of_gens.append(deepcopy(best))
    self.best_fitness.append(best.fitness)
    self.average_of_gens.append(average)
    self.diversity_of_fitness.append(variation)

  def _perform_generation(self):
    """
    Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
    """
    # mutate every individual
    if self.optimization_behavior == OptimizationBehavior.ALL_INDIVIDUALS:
      self.population = [self._optimize(ind, self.steps) for ind in self.population]

    # mutate sampled individuals
    if self.optimization_behavior == OptimizationBehavior.SAMPLED_INDIVIDUALS:
      sample_idxs = np.random.choice(np.arange(self.pop_size), self.pop_size//4, replace=False)
      sampled_individuals = [self._optimize(self.population[ind], self.steps*4) for ind in sample_idxs]
      for (idx, ind) in zip(sample_idxs, sampled_individuals):
        self.population[idx] = ind

    # select promising parents
    sel_fun = self.selection["fun"]

    parents = sel_fun(self.population, self.pop_size, **self.selection["kwargs"])

    #mutate parents
    if self.optimization_behavior == OptimizationBehavior.ALL_PARENTS:
      parents = [self._optimize(parent, self.steps) for parent in parents]

    #mutate a sample of parents
    if self.optimization_behavior == OptimizationBehavior.SAMPLED_PARENTS:
      sample_idxs = np.random.choice(np.arange(self.pop_size), self.pop_size//4, replace=False)
      sampled_individuals = [self._optimize(parents[ind], self.steps*4) for ind in sample_idxs]
      for (idx, ind) in zip(sample_idxs, sampled_individuals):
        parents[idx] = ind

    # generate offspring
    offspring_population = Parallel(n_jobs=self.n_jobs)(delayed(generate_offspring)
      (t, self.crossovers, self.mutations, self.coeff_opts,
      parents, self.internal_nodes, self.leaf_nodes,
      constraints={"max_tree_size": self.max_tree_size})
      for t in parents)

    #mutate all offspring
    if self.optimization_behavior == OptimizationBehavior.ALL_OFFSPRING:
      offspring_population = [self._optimize(pop, self.steps) for pop in offspring_population]

    #mutate sampled offspring
    if self.optimization_behavior == OptimizationBehavior.SAMPLED_OFFSPRING:
      sample_idxs = np.random.choice(np.arange(self.pop_size), self.pop_size//4, replace=False)
      sampled_individuals = [self._optimize(offspring_population[ind], self.steps*4) for ind in sample_idxs]
      for (idx, ind) in zip(sample_idxs, sampled_individuals):
        offspring_population[idx] = ind

    # evaluate each offspring and store its fitness
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in offspring_population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory + self.memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      offspring_population[i].fitness = fitnesses[i]
    # store cost
    self.num_evals += self.pop_size
    # update the population for the next iteration
    self.population = offspring_population
    # update info
    self.num_gens += 1
    fitness_list = [t.fitness for t in self.population]
    best = self.population[np.argmax(fitness_list)]
    average = np.mean(fitness_list)
    variation = np.std(fitness_list)
    diversity_gen = compute_all_pairwise_distance(self.population)

    self.diversity_of_population.append(diversity_gen)
    self.best_of_gens.append(deepcopy(best))
    self.best_fitness.append(best.fitness)
    self.average_of_gens.append(average)
    self.diversity_of_fitness.append(variation)

  def _optimize(self, individual, steps):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    constants = individual.get_subtrees_consts()
    if len(constants)>0:
        optimizer = optim.AdamW(constants, lr=self.lr, amsgrad=True)

    for _ in range(steps):

        if len(constants)>0 and len(self.memory)>self.batch_size:
            target_tree = copy.deepcopy(individual)

            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = individual.get_output_pt(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(self.batch_size, dtype=torch.float)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_tree.get_output_pt(non_final_next_states).max(1)[0].float()

            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(constants, 100)
            optimizer.step()
    return individual

  def evolve(self):
    """
    Runs the evolution until a termination criterion is met;
    first, a random population is initialized, second the generational loop is started:
    every generation, promising parents are selected, offspring are generated from those parents,
    and the offspring population is used to form the population for the next generation
    """
    # set the start time
    self.start_time = time.time()
    self.optimize_count = 0
    self._initialize_population()

    # generational loop
    while not self._must_terminate():
      # perform one generation
      self._perform_generation()
      # log info
      if self.verbose:
        print("gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {}".format(
            self.num_gens, self.best_of_gens[-1].fitness, len(self.best_of_gens[-1])
            ))
