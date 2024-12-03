import random
import numpy as np
import gym
import compiler_gym
from typing import List, Tuple
from absl import app, flags
import statistics


flags.DEFINE_list(
    "flags",
    [
        "-add-discriminators",
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
        "-cross-dso-cfi",
        "-deadargelim",
        "-dse",
        "-reg2mem",
        "-div-rem-pairs",
        "-early-cse-memssa",
        "-elim-avail-extern",
        "-ee-instrument",
        "-flattencfg",
        "-float2int",
        "-forceattrs",
        "-inline",
        "-insert-gcov-profiling",
        "-gvn-hoist",
        "-gvn",
        "-globaldce",
        "-globalopt",
        "-globalsplit",
        "-guard-widening",
        "-hotcoldsplit",
        "-ipconstprop",
        "-ipsccp",
        "-indvars",
        "-irce",
        "-infer-address-spaces",
        "-inferattrs",
        "-inject-tli-mappings",
        "-instsimplify",
        "-instcombine",
        "-instnamer",
        "-jump-threading",
        "-lcssa",
        "-licm",
        "-libcalls-shrinkwrap",
        "-load-store-vectorizer",
        "-loop-data-prefetch",
        "-loop-deletion",
        "-loop-distribute",
        "-loop-fusion",
        "-loop-guard-widening",
        "-loop-idiom",
        "-loop-interchange",
        "-loop-load-elim",
        "-loop-predication",
        "-loop-reroll",
        "-loop-rotate",
        "-loop-simplifycfg",
        "-loop-simplify",
        "-loop-sink",
        "-loop-reduce",
        "-loop-unroll-and-jam",
        "-loop-versioning-licm",
        "-loop-versioning",
        "-loweratomic",
        "-lower-constant-intrinsics",
        "-lower-expect",
        "-lower-guard-intrinsic",
        "-lowerinvoke",
        "-lower-matrix-intrinsics",
        "-lowerswitch",
        "-lower-widenable-condition",
        "-memcpyopt",
        "-mergefunc",
        "-mergeicmps",
        "-mldst-motion",
        "-sancov",
        "-name-anon-globals",
        "-nary-reassociate",
        "-newgvn",
        "-pgo-memop-opt",
        "-partial-inliner",
        "-partially-inline-libcalls",
        "-post-inline-ee-instrument",
        "-functionattrs",
        "-mem2reg",
        "-prune-eh",
        "-reassociate",
        "-redundant-dbg-inst-elim",
        "-rpo-functionattrs",
        "-rewrite-statepoints-for-gc",
        "-sccp",
        "-slp-vectorizer",
        "-sroa",
        "-scalarizer",
        "-separate-const-offset-from-gep",
        "-simple-loop-unswitch",
        "-sink",
        "-speculative-execution",
        "-slsr",
        "-strip-dead-prototypes",
        "-strip-debug-declare",
        "-strip-nondebug",
        "-strip",
        "-tailcallelim",
        "-mergereturn"
    ],
    "List of optimizations to explore.",
)
flags.DEFINE_integer("population_size", 10, "Number of individuals in the population.")
flags.DEFINE_integer("generation_count", 15, "Number of generations to evolve.")
flags.DEFINE_integer("episode_len", 5, "Length of each sequence of optimizations.")
flags.DEFINE_float("mutation_rate", .1, "Probability of mutation.")
flags.DEFINE_float("crossover_rate", .8, "Probability of crossover.")
flags.DEFINE_integer("iterations", 3, "Training")


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
        print(f"Generation {generation + 1}: Best Fitness = {current_best_fitness:.4f}")
        new_population = []
        while len(new_population) < FLAGS.population_size:
            parent_indices = random.choices(range(len(population)), weights=fitnesses, k=2)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:FLAGS.population_size]
    # print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")
    print(f"Achieved Fitness: {best_fitness}, Optimal Sequence: {best_individual}")

    return best_fitness, best_individual

def main(argv):
    del argv
    env = compiler_gym.make("llvm-v0")
    
    #without benchmarks
    #env.reset() 

    #if using benchmarks
    #benchmarks = ["benchmark://cbench-v1/crc32","benchmark://cbench-v1/dijkstra","benchmark://cbench-v1/bzip2","benchmark://cbench-v1/jpeg-c"] #add additional
    benchmarks = ["benchmark://cbench-v1/crc32"] #add additional

    for benchmark in benchmarks:
        print(f"Running Benchmark: {benchmark}")
        env.reset(benchmark=benchmark)
    
        print(f"Episode length: {FLAGS.episode_len}")
        print(f"Population Size: {FLAGS.population_size}")
        print(f"Generation Count: {FLAGS.generation_count}")
        print(f"Mutation Rate: {FLAGS.mutation_rate}")
        print(f"Crossover Rate: {FLAGS.crossover_rate}")
        print(f"Iterations: {FLAGS.iterations}")
        print(f"Observations: Runtime, IR Instruction Count, Autophase Instruction Count")
        print(f"Action space: {FLAGS.flags}\n")
        

        if FLAGS.iterations == 1:
            genetic_algorithm(env)
            return

        best_fitness = []
        best_individual = []
        for t in range(1, FLAGS.iterations + 1):
            print("Iteration", t, " of ", FLAGS.iterations)
            fitness, individual = genetic_algorithm(env)
            best_fitness.append(fitness)
            best_individual.append(individual)
        print("\nGenetic Algorithm Performance Review w/ Multiple Iterations")
        print(f"Algorthm Fitness Results: {best_fitness}\n")
        print(f"Best Fitness: {max(best_fitness)}\n")
        print(f"Avg Fitness: {statistics.mean(best_fitness)}\n")
        print(f"Worst Fitness: {min(best_fitness)}\n")
        print(f"Best Inviduals: {best_individual}")
        print("--------------------------------------")
        env.close()

if __name__ == "__main__":
    app.run(main)
