import random
import numpy as np
import gym
import compiler_gym
from typing import List, Tuple
from absl import app, flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

flags.DEFINE_list(
    "flags",
    [ "-add-discriminators",
    "-adce",
    "-loop-unroll",
    "-loop-unswitch",
    "-loop-vectorize",
    "-aggressive-instcombine",
    "-alignment-from-assumptions",
    "-always-inline",
    "-argpromotion",
    "-attributor",
    "-barrier",
    "-bdce",
    "-loop-instsimplify",
    "-break-crit-edges",
    "-simplifycfg",
    "-dce",
    "-called-value-propagation",
    "-die",
    "-canonicalize-aliases",
    "-consthoist",
    "-constmerge",
    "-constprop",
    "-coro-cleanup",
    "-coro-early",
    "-coro-elide",
    "-coro-split",
    "-correlated-propagation",
    "-cross-dso-cfi"],
    "List of optimizations to explore.",
)
flags.DEFINE_integer("population_size", 10, "Number of individuals in the population.")
flags.DEFINE_integer("generation_count", 5, "Number of generations to evolve.")
flags.DEFINE_integer("episode_len", 5, "Length of each sequence of optimizations.")
flags.DEFINE_float("mutation_rate", .1, "Probability of mutation.")
flags.DEFINE_float("crossover_rate", .8, "Probability of crossover.")

FLAGS = flags.FLAGS

def evaluate_fitness(env, individual: List[str]) -> float:
    env.reset()
    total_reward = 0
    initial_ic = env.observation["IrInstructionCount"]
    initial_rt = env.observation["Runtime"][0]
    initial_auto = env.observation["Autophase"][51]

    for action in individual:
        action_index = env.action_space.flags.index(action)
        observation, reward, done, info = env.step(action_index)
        combined = rewards(env, initial_rt, initial_ic, initial_auto)
        total_reward += combined if combined is not None else 0
        if done:
            break
        
    return total_reward

def rewards(env, initial_rt, initial_ic, inital_auto):
    after_ic = env.observation["IrInstructionCount"]
    after_rt = env.observation["Runtime"][-1]
    runtime = initial_rt - after_rt
    if runtime < 0:
        runtime = 0
    runtime *= 0.5
    ic = initial_ic - after_ic
    if ic < 0:
        ic = 0
    ic *= 0.003
    after_auto = env.observation["Autophase"][51]
    auto = inital_auto - after_auto
    if auto < 0:
        auto = 0
    auto *= 0.002
    combined = runtime + ic + auto
    return combined

def crossover(parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
    if random.random() < FLAGS.crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        child1, child2 = parent1[:], parent2[:]
    return child1, child2

def mutate(individual: List[str]) -> List[str]:
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < FLAGS.mutation_rate:
            mutated_individual[i] = random.choice(FLAGS.flags)
    return mutated_individual

def genetic_algorithm(env):
    population = [random.choices(FLAGS.flags, k=FLAGS.episode_len) for _ in range(FLAGS.population_size)]
    best_fitness = float('-inf')
    best_individual = None
    for generation in range(FLAGS.generation_count):
        fitnesses = [evaluate_fitness(env, individual) for individual in population]
        current_best_fitness = max(fitnesses)
        current_best_individual = population[np.argmax(fitnesses)]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
        print(f"Generation {generation + 1}: Best Fitness = {current_best_fitness}")
        new_population = []
        while len(new_population) < FLAGS.population_size:
            parent_indices = random.choices(range(len(population)), weights=fitnesses, k=2)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:FLAGS.population_size]
    print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")
    return best_individual, best_fitness

def main(argv):
    del argv
    env = compiler_gym.make("llvm-v0")
    env.reset()
    best_individual, best_fitness = genetic_algorithm(env)
    print(f"Optimal Sequence: {best_individual}, Achieved Fitness: {best_fitness}")
    env.close()

if __name__ == "__main__":
    app.run(main)
