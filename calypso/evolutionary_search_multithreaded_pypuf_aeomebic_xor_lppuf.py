from pypuf.simulation import XORArbiterPUF, InterposePUF
from pypuf.io import random_inputs
import random
import numpy as np
import math
import copy
import argparse
from multiprocessing import Process, Value
import multiprocessing
import pandas as pd
from numpy.random import default_rng
from pypuf.metrics import uniqueness, uniqueness_data
import pypuf.metrics
import lppuf
from numpy.random import seed
from datetime import datetime

parser = argparse.ArgumentParser(description ='Evolutionary algorithm based attack on PUFs.')
parser.add_argument('--target-degree', help="The number of XORs to be used in the XOR arbiter PUF", required=True)
parser.add_argument('--cut-length', help="The cut length to use", required=True)
parser.add_argument('--challenge-length', help="Number of challenges to consider", required=True)
args = parser.parse_args()

GENERATION = 1
SPLIT_LENGTH = 4
SPLIT_VECTOR = []
for index in range(SPLIT_LENGTH):
    SPLIT_VECTOR.append(2 ** index)
SPLIT_VECTOR = np.array(SPLIT_VECTOR)
CHALLENGE_NUM = int(args.challenge_length)
CHALLENGE_LENGTH = 64
PUF_LENGTH = int(args.target_degree)
CUT_LENGTH = int(args.cut_length)

NUM_SAMPLES = CHALLENGE_NUM // SPLIT_LENGTH
INITIAL_NUM_SAMPLES = CHALLENGE_NUM

NEW_CHALLENGE_SET_HIGH_ = Value('f', 0)
process_pool_leader = {}
POPULATION_SIZE = 1
PROC = 1

class Chromosome():
    def __init__(self, n, k, external_weights=None, external_biases=None):
        self.weight_vector = None
        self.bias_vector = None
        if(external_weights is None):
            self.weight_vector = np.random.normal(loc=0, scale=0.5, size=(k, n))
        else:
            self.weight_vector = external_weights

        if(external_biases is None):
            self.bias_vector = np.random.normal(loc=0, scale=0.5, size=(k,))
        else:
            self.bias_vector = external_biases

        self.generation = 0
        self.puf = None
        self.fitness = 0
        self.age = 0
        self.standardized_fitness = 0
        self.mated = 0
        self.crossover_indexes = []
        self.stage_1_crossover_indexes = []
        self.stage_2_crossover_indexes = []
        self.bias = 0


    def set_generation(self, generation):
        self.generation = generation

    def print_parameters(self):
        print(self.weight_vector)
        print(self.bias_vector)

    def generate_puf(self):
        if(self.puf is not None):
            del self.puf
        self.puf = XORArbiterPUF(n=CHALLENGE_LENGTH, k=PUF_LENGTH, noisiness=0,
                        external_weights=self.weight_vector, external_biases=self.bias_vector)

    def evaluate_puf_fitness(self, golden_challenge_set, target_responses, optimal_bias):
        self.generate_puf()
        predicted_responses = self.puf.eval(golden_challenge_set)
        matches = np.sum(predicted_responses == target_responses)
        bias = pypuf.metrics.bias_data(predicted_responses) 
        return matches/len(predicted_responses) #- 10 * np.abs(optimal_bias - bias)

class GeneticAlgoWrapper:
    def __init__(self, targetPUF, challenge_set, response_golden_set, n, k, target_response_set, optimal_bias,
            test_challenges, test_responses):
        self.max_population_size = POPULATION_SIZE
        self.initial_max_population_size = self.max_population_size
        self.n = n
        self.k = k
        self.targetPUF = targetPUF
        self.population = []
        self.golden_challenge_set = challenge_set
        self.golden_response_set = response_golden_set
        self.target_responses = target_response_set
        self.randomness_in_population = 20
        self.last_new_find = 0
        self.last_new_fitter_population = 0
        self.mutation_std = 0.5
        self.MODE = "normal"
        self.sub_puf_length = 16
        self.sub_puf_index = (CHALLENGE_LENGTH // self.sub_puf_length) - 1
        self.mutation_round_robin_index = 0
        self.optimal_bias = optimal_bias
        self.crossover_indexes = [] 
        self.new_challenge_set = test_challenges
        self.new_response_set = test_responses

    def compare_initial_population(self):
        # Use this function to convince yourself
        # that the initial population is decently different
        pass

    def generate_initial_population(self, population=None):
        for _ in range(self.max_population_size):
            c = Chromosome(self.n, self.k)
            c.generate_puf()
            if(population is not None):
                population.append(c)
        self.compare_initial_population()
        return population

    def evaluate_subset_fitness_euclidean_distance(self, member):
        predicted_responses = member.puf.eval(self.golden_challenge_set)
        evaluated_vector = np.array([])
        member.age = member.age + 1
        for index in range(len(self.golden_challenge_set) // SPLIT_LENGTH):
            split = predicted_responses[index * SPLIT_LENGTH : ((index+1) * SPLIT_LENGTH)]
            split[split == -1] = 0
            split = np.array(split)
            split = split.dot(SPLIT_VECTOR)
            evaluated_vector= np.append(evaluated_vector, split)
        evaluated_vector = evaluated_vector.astype(float)
        self.golden_response_set = self.golden_response_set.astype(float)
        norm = np.linalg.norm(evaluated_vector - self.golden_response_set)
        member.fitness = -1 * norm
        return member

    def evaluate_subset_fitness_jaccard(self, member):
        predicted_responses = member.puf.eval(self.golden_challenge_set)
        ands = 0
        ors = 0
        for index in range(len(predicted_responses)):
            if(predicted_responses[index] == self.target_responses[index]):
                ands = ands + 1
        member.fitness = ands / (2 * len(predicted_responses) - ands)
        member.age = member.age + 1
        member.bias = pypuf.metrics.bias_data(predicted_responses)
        #member.fitness = member.fitness + 0.001 * member.age
        return member

    def evaluate_subset_fitness(self, member):
        
        predicted_responses = member.puf.eval(self.golden_challenge_set)
        matches = np.sum(predicted_responses == self.target_responses)
        member.age = member.age + 1
        member.bias = pypuf.metrics.bias_data(predicted_responses)
        member.fitness = matches/len(predicted_responses)

    def evaluate_population_fitness(self, population):
        if(population is None):
            print("None")
        for member in population:
            self.evaluate_subset_fitness(member)
        return population

    def sort_population_members(self, population):
        population.sort(key=lambda x: x.fitness)
        population.reverse()
        return population

    def standardize_fitness(self):
        fitnesses = []
        for member in self.population:
            fitnesses.append(member.fitness)
        fitnesses = np.array(fitnesses)
        logger = open("logger", "a")
        logger.write(str(fitnesses))

        min_fitness = np.min(fitnesses)
        max_fitness = np.max(fitnesses)
        fitness_sum = np.sum(fitnesses)
        standarized_fitness = []
        for member in self.population:
            member.standardized_fitness = (member.fitness - min_fitness) / (max_fitness - min_fitness)
            standarized_fitness.append(member.standardized_fitness)

        logger.write("\n")
        logger.write(str(standarized_fitness))
        logger.write("\n#################\n")
        logger.close()

    def roulette_selection(self):
        while True:
            index = random.randint(0, len(self.population) - 1)
            if(self.population[index].standardized_fitness > random.random()):
                return index

    def reproduce(self):
        children = []
        population_length = len(self.population)
        PUFS_TO_SWAP = random.randint(0, CUT_LENGTH - 1)
        swap_list = []
        for _ in range(PUFS_TO_SWAP):
            if(len(self.crossover_indexes) == 0):
                for i in range(self.k * self.n):
                    self.crossover_indexes.append(i)
            swap_list.append(self.crossover_indexes.pop(random.randint(0, len(self.crossover_indexes) - 1)))

        bias_PUFS_TO_SWAP = 1 
        bias_swap_list = []
        for _ in range(bias_PUFS_TO_SWAP):
            bias_swap_list.append(random.randint(0, self.k - 1))

        CROSSOVER_PROBABILITY = 1
        mating_pool = self.population
        for _ in range(int(CROSSOVER_PROBABILITY * len(mating_pool))):
            member_index = 0
            spouse_index = 0

            while(member_index == spouse_index):                                
                member_index = random.randint(0, len(self.population) - 1)
                spouse_index = random.randint(0, len(self.population) - 1)                                                

            member = copy.deepcopy(mating_pool[member_index])
            spouse = copy.deepcopy(mating_pool[spouse_index])
            member.mated = 1
            mating_pool[spouse_index].mated = 1

            w1 = member.weight_vector.reshape((self.k * self.n))
            b1 = member.bias_vector
            w2 = spouse.weight_vector.reshape((self.k * self.n))
            b2 = spouse.bias_vector
            weight_vector_split = math.ceil((self.k * self.n)/2)
            bias_vector_split = math.ceil(self.k/2)

            child_1_weights = np.array([])
            child_2_weights = np.array([])
            child_1_bias = np.array([])
            child_2_bias = np.array([])

            SWAP_LENGTH = 1
            run_length = (self.n * self.k) // SWAP_LENGTH
            block_size = SWAP_LENGTH
            if(self.MODE == "streamlined"):
                pass

            for i in range(run_length):
                if(i not in swap_list):
                    child_1_weights = np.append(child_1_weights, w1[i * block_size : (i+1) * block_size])
                else:
                    child_1_weights = np.append(child_1_weights, w2[i * block_size : (i+1) * block_size])

            for i in range(run_length):
                if(i not in swap_list):
                    child_2_weights = np.append(child_2_weights, w2[i * block_size : (i+1) * block_size])
                else:
                    child_2_weights = np.append(child_2_weights, w1[i * block_size : (i+1) * block_size])

            # bias
            for i in range(self.k):
                if(i not in bias_swap_list):
                    child_1_bias = np.append(child_1_bias, b1[i])
                else:
                    child_1_bias = np.append(child_1_bias, b2[i])

            for i in range(self.k):
                if(i not in bias_swap_list):
                    child_2_bias = np.append(child_2_bias, b2[i])
                else:
                    child_2_bias = np.append(child_2_bias, b1[i])

            child_1_weights = child_1_weights.reshape((self.k, self.n))
            child_2_weights = child_2_weights.reshape((self.k, self.n))
            child_1_bias = child_1_bias.reshape((self.k, ))
            child_2_bias = child_2_bias.reshape((self.k, ))

            child_1 = Chromosome(self.n, self.k, child_1_weights, child_1_bias)
            child_2 = Chromosome(self.n, self.k, child_2_weights, child_2_bias)

            child_1.set_generation(GENERATION)
            child_2.set_generation(GENERATION)

            child_1.age = 0
            child_2.age = 0

            children.append(child_1)
            children.append(child_2)

        return children

    def reproduce_xor(self):
        children = []
        population_length = len(self.population)
        CROSSOVER_PROBABILITY = 20
        for _ in range(int(population_length * CROSSOVER_PROBABILITY) // 2):
            member_index = 0
            spouse_index = 0

            while(member_index == spouse_index):
                member_index = self.roulette_selection()
                spouse_index = self.roulette_selection()

            member = copy.deepcopy(self.population[member_index])
            spouse = copy.deepcopy(self.population[spouse_index])
            self.population[member_index].mated = 1
            self.population[spouse_index].mated = 1

            child_weights = member.weight_vector + spouse.weight_vector
            child_bias = member.bias_vector + spouse.bias_vector
            child = Chromosome(self.n, self.k, child_weights, child_bias)
            child.set_generation(GENERATION)
            child.age = 0
            children.append(child)
        return children


    def aeomebic_reproduce(self):
        children = []
        repetitions = 1
        population_backup = copy.deepcopy(self.population)
        while population_backup:
            member = population_backup.pop(0)
            child = Chromosome(self.n, self.k, member.weight_vector, member.bias_vector)
            child.set_generation(GENERATION)
            child.age = 0
            child.generate_puf()
            children.append(child)
            del member
        return children

    def mutate_children(self, children):
        self.mutation_std = 0.2
        mean = 0

        weight_mutation_rate = 0.1
        bias_mutation_rate =  1 / (self.k)

        if(self.MODE == "streamlined"):
            pass

        for child in children:
            child_weights = child.weight_vector.reshape((self.k * self.n))
            if(random.random() > 0.1):
                continue
            for _ in range(int(self.n * self.k * weight_mutation_rate)):
                index = CHALLENGE_LENGTH * random.randint(0, self.k - 1) +  random.randint((self.sub_puf_index) * self.sub_puf_length, ((self.sub_puf_index + 1) * self.sub_puf_length) - 1)
                child_weights[index] = child_weights[index] + np.random.normal(loc=mean, scale=self.mutation_std)
            child.weight_vector = child_weights.reshape((self.k, self.n))
            for index in range(int(self.k * bias_mutation_rate)):
                index = random.randint(0, self.k - 1)
                child.bias_vector[index] = child.bias_vector[index] + np.random.normal(loc=mean, scale=self.mutation_std)
        return children

    def mutate_subset(self, child):
        mutation_rate =  [1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        mean = 0
        bias_mutation_rate =  1 / (self.k)

        stage_1_mutation_indices = []
        indices_to_change = CUT_LENGTH
        bias_index = 0
        for rate in mutation_rate:
            self.mutation_std = rate
            for _ in range(indices_to_change):
                #if(len(child.stage_1_crossover_indexes) == 0):
                #    for j in range(self.n * self.k):
                #        child.stage_1_crossover_indexes.append(j)
                #stage_1_mutation_indices.append(child.stage_1_crossover_indexes.pop(
                #    random.randint(0, len(child.stage_1_crossover_indexes) - 1)))
                stage_1_mutation_indices.append(random.randint(0, (self.n * self.k) - 1))

                original_weight_vector = copy.deepcopy(child.weight_vector)
                original_bias_vector = copy.deepcopy(child.bias_vector)
                child.weight_vector = child.weight_vector.reshape((self.k * self.n))
                for index in stage_1_mutation_indices:
                    child.weight_vector[index] = child.weight_vector[index] + np.random.normal(loc=0, scale=self.mutation_std)
                child.weight_vector = child.weight_vector.reshape((self.k, self.n))
                
                index = (bias_index + 1) % self.k #random.randint(0, self.k - 1)
                bias_index = bias_index + 1
                child.bias_vector[index] = child.bias_vector[index] + np.random.normal(loc=0, scale=self.mutation_std)

                evaluated_fitness = child.evaluate_puf_fitness(self.golden_challenge_set, self.target_responses, self.optimal_bias)
                if(evaluated_fitness <= child.fitness):
                    child.weight_vector = original_weight_vector
                    child.bias_vector = original_bias_vector
                    child.generate_puf()
                else:
                    child.fitness = evaluated_fitness
                    child.age = 0
                    return child
        return child

    def mutate_children_round_robin(self, children):
        for child in children:
            self.mutate_subset(child)
    
    def add_to_population(self, children):
        for child in children:
            child.generate_puf()
            self.population.append(child)

    def kill_weak_children(self):
        population_size = len(self.population)
        if(population_size <= self.max_population_size):
            return

        OLD_AGE = 10
        evictions = len(self.population) - self.max_population_size
        while(len(self.population) > self.max_population_size):
            weak_member = self.population.pop(self.max_population_size)
            del weak_member

    def add_new_members(self):
        NEW_MEMBERS = int(0.2 * POPULATION_SIZE)
        for _ in range(NEW_MEMBERS):
            c = Chromosome(self.n, self.k)
            c.generate_puf()
            self.population.append(c)

    def compute_average_population_fitness(self):
        total_fitness = 0
        f = []
        for member in self.population:
            total_fitness = total_fitness + member.fitness
            f.append(member.fitness)
        return total_fitness/len(self.population)

    def count_young_population(self):
        total_youth = 0
        for member in self.population:
            if(member.age <= 1):
                total_youth = total_youth + 1
        return total_youth

    def compute_average_population_age(self):
        total_age = 0
        totals = []
        for member in self.population:
            total_age = total_age + member.age
            totals.append(member.age)
        return total_age / len(self.population)

    def generate_response_golden_set(self):
        self.golden_response_set = np.array([])
        for index in range(len(self.golden_challenge_set) // SPLIT_LENGTH):
            split = self.target_responses[index * SPLIT_LENGTH : ((index+1) * SPLIT_LENGTH)]
            split[split == -1] = 0
            split = np.array(split)
            split = split.dot(SPLIT_VECTOR)
            self.golden_response_set = np.append(self.golden_response_set, split)

    def randomize_dataset(self):
        self.golden_challenge_set = random_inputs(n=CHALLENGE_LENGTH, N=CHALLENGE_NUM, seed=random.randint(0, 60000))
        self.target_responses = self.targetPUF.eval(self.golden_challenge_set)

    def evaluate_pop_bias(self, population):
        total_bias = 0
        b = []
        for member in population:
            total_bias = total_bias + member.bias
            b.append(np.abs(member.bias))
        return total_bias / len(population)

    def compute_majority_voting(self, population, test_challenges, test_responses):
        majority_vote = np.array([0] * len(test_responses))
        for index in range(int(len(population))):            
            pred = population[index].puf.eval(test_challenges)
            majority_vote = majority_vote + pred
            break

        majority_vote[majority_vote < 0] = -1
        majority_vote[majority_vote > 0] = 1
        majority_vote[majority_vote == 0] = 1
        return np.sum(majority_vote == test_responses) / len(test_challenges)

    def attack(self):
        global GENERATION
        global GENERATIONAL_HIGH_
        global NEW_CHALLENGE_SET_HIGH_
        global f
        delta = 10 
        max_observed_fitness =  0 
        max_population_fitness = 0 
        population = []
        population = self.generate_initial_population(population)
        best_solution = copy.deepcopy(population[0])
        previous_pop_fitness = 0
        last_new_find_fitness = 0
        last_new_find_index = 0
        new_challenge_set_high = 0
        while True:
            random.seed(datetime.now().timestamp())
            self.evaluate_population_fitness(population)
            self.sort_population_members(population)
            process_pool_leader[multiprocessing.current_process().pid] = population[0]
            max_fitness = np.sum(self.target_responses == population[0].puf.eval(self.golden_challenge_set))/len(self.golden_challenge_set)
            max_fitness_iteration = np.sum(self.target_responses == population[0].puf.eval(self.golden_challenge_set))/len(self.golden_challenge_set)
            if(GENERATION):
                max_observed_fitness = max_fitness_iteration
                prev_challenge_set_high = new_challenge_set_high
                new_challenge_set_high = self.compute_majority_voting(population, self.new_challenge_set, self.new_response_set)
                movement = ""
                if(new_challenge_set_high > prev_challenge_set_high):
                    movement = "+"

                if(NEW_CHALLENGE_SET_HIGH_.value < new_challenge_set_high):
                    with NEW_CHALLENGE_SET_HIGH_.get_lock():
                        majority_vote = np.array([0] * len(self.new_response_set))
                        for index in process_pool_leader.keys():
                            pred = process_pool_leader[index].puf.eval(self.new_challenge_set)
                            majority_vote = majority_vote + pred
                        majority_vote[majority_vote < 0] = -1
                        majority_vote[majority_vote > 0] = 1
                        majority_vote[majority_vote == 0] = 1
                        NEW_CHALLENGE_SET_HIGH_.value = np.sum(majority_vote == self.new_response_set) / len(self.new_response_set)
                
                if(max_fitness > last_new_find_fitness):
                    last_new_find_fitness = max_fitness
                    last_new_find_index = GENERATION
                    print("[!!!] Test acc: ", prev_challenge_set_high, " --> ", new_challenge_set_high, "(", movement, ")", ". Gen high: ", max_fitness, " (gen: ", GENERATION, ", Last find: ", last_new_find_index, "). Bias: ", round(self.evaluate_pop_bias(population),4), ". Max: ", round(NEW_CHALLENGE_SET_HIGH_.value, 4), ", opt. bias: ", self.optimal_bias, "ID: ", multiprocessing.current_process().pid)                
                
                self.last_new_find = -1
                self.last_new_fitter_population = 1
                self.max_population_size = self.initial_max_population_size
                if(np.abs(self.evaluate_pop_bias(population)) > 0.25 and GENERATION > 50):
                    with NEW_CHALLENGE_SET_HIGH_.get_lock():
                        NEW_CHALLENGE_SET_HIGH_.value = 0
                    return 
            GENERATION = GENERATION + 1
            self.last_new_find = self.last_new_find + 1
            self.mutate_children_round_robin(population)


    def evolutionary_attack(self):
        for _ in range(80):
            process = multiprocessing.Process(target=self.attack)
            process.start() 

challenges = challenges=random_inputs(n=CHALLENGE_LENGTH, N=CHALLENGE_NUM, seed=random.randint(0,100))
targetPUF = lppuf.LPPUFv1(n=CHALLENGE_LENGTH, m=PUF_LENGTH, seed=random.randint(0,100))
responses = targetPUF.eval(challenges)
test_challenges = random_inputs(n=CHALLENGE_LENGTH, N=1000, seed=random.randint(0,100))
test_responses = targetPUF.eval(test_challenges)
optimal_bias = pypuf.metrics.bias_data(responses)
print("Optimal bias: ", optimal_bias)

geneticAlgoWrapper = GeneticAlgoWrapper(None, challenges, None, CHALLENGE_LENGTH, PUF_LENGTH, responses, optimal_bias,
        test_challenges, test_responses)
geneticAlgoWrapper.evolutionary_attack()
