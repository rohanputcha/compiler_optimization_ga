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
    # [
    #     "-break-crit-edges",
    #     "-early-cse-memssa",
    #     "-gvn-hoist",
    #     "-gvn",
    #     "-instcombine",
    #     "-instsimplify",
    #     "-jump-threading",
    #     "-loop-reduce",
    #     "-loop-rotate",
    #     "-loop-versioning",
    #     "-mem2reg",
    #     "-newgvn",
    #     "-reg2mem",
    #     "-simplifycfg",
    #     "-sroa",
    # ],
    "List of optimizations to explore.",
)
flags.DEFINE_integer("population_size", 10, "Number of individuals in the population.")
flags.DEFINE_integer("generation_count",5, "Number of generations to evolve.")
flags.DEFINE_integer("episode_len", 5, "Length of each sequence of optimizations.")
flags.DEFINE_float("mutation_rate", .1, "Probability of mutation.")
flags.DEFINE_float("crossover_rate", .8, "Probability of crossover.")

FLAGS = flags.FLAGS

# Helper function: Generate a random individual (sequence of passes)
#def generate_individual() -> List[str]:


#fitness evaluation --> runtime, IRIC, Autophase IC
def evaluate_fitness(env, individual: List[str]) -> float:
    env.reset()
    total_reward = 0
    initial_ic = env.observation["IrInstructionCount"]
    initial_rt=env.observation["Runtime"][0]
    initial_auto= env.observation["Autophase"][51]

    for action in individual:
        action_index = env.action_space.flags.index(action)
        observation, reward, done, info = env.step(action_index)
        combined= rewards(env, initial_rt, initial_ic, initial_auto)

        # print(f"Action: {action}, Observation: {observation}, Reward: {combined}, Done: {done}, Info: {info}")
        total_reward += combined if combined is not None else 0
        if done:
            break
        
    return total_reward

# Reward Calculation
def rewards(env, initial_rt, initial_ic, inital_auto):
    after_ic = env.observation["IrInstructionCount"]
    after_rt = env.observation["Runtime"][-1]
    
    #runtime
    runtime=initial_rt-after_rt
    if runtime <0:
        runtime=0
    runtime = runtime * .5

    #instruction cost
    ic=initial_ic-after_ic
    # if ic>0:
    #     # print("the ic is not negative")
    if ic < 0:
        ic=0
    ic = ic * .003

    after_auto = env.observation["Autophase"][51]
    # print("the iauto is: ", inital_auto)
    # print("the Aauto is: ", after_auto)
    auto = inital_auto - after_auto 
    # print("the auto is: ", auto)
    if auto< 0:
        auto = 0
    auto = auto * .002

    combined = runtime + ic + auto

    # print(combined, "runtime: ", runtime, " ic: ", ic, " auto: ", auto)# " cache hits: ", hits, " cache misses: ", misses)
    return combined


# Helper function: Perform crossover between two individuals
#def crossover(parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
   

# Helper function: Mutate an individual
def mutate(individual: List[str]) -> List[str]:
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < FLAGS.mutation_rate:
            mutated_individual[i] = random.choice(FLAGS.flags)
    return mutated_individual


# Genetic Algorithm Implementation
#def genetic_algorithm(env):
        #print(f"Generation {generation}, Best Fitness: {best_fitness}", " Best idividual:", [best_individual], "\n")

    #print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")



def main(argv):
    del argv  # Unused
    # FLAGS.env="llvm-v0"
    env = compiler_gym.make("llvm-v0")
    env.reset()
    


    #genetic_algorithm(env)
    env.close()
if __name__ == "__main__":
    app.run(main)
