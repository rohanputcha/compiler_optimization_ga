import random
import statistics
from collections import namedtuple
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags
from torch.distributions import Categorical

import compiler_gym.util.flags.episodes  # noqa Flag definition.  <-- number of episodes
import compiler_gym.util.flags.learning_rate  # noqa Flag definition.
import compiler_gym.util.flags.seed  # noqa Flag definition.
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit

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

flags.DEFINE_integer("episode_len", 5, "Number of transitions per episode.")
flags.DEFINE_integer("hidden_size", 64, "Latent vector size.")
flags.DEFINE_integer("log_interval", 4, "Episodes per log output.")
flags.DEFINE_integer("episodes_count",240, "Number of episodes.")
flags.DEFINE_integer("iterations", 24, "Times to redo entire training.")
flags.DEFINE_float("exploration", 0.0, "Rate to explore random transitions.")
flags.DEFINE_float("mean_smoothing", 0.95, "Smoothing factor for mean normalization.")
flags.DEFINE_float("std_smoothing", 0.4, "Smoothing factor for std dev normalization.")

eps = np.finfo(np.float32).eps.item()

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

FLAGS = flags.FLAGS


class MovingExponentialAverage:
   
    def __init__(self, smoothing_factor):
        self.smoothing_factor = smoothing_factor
        self.value = None

    def next(self, entry):
        assert entry is not None
        if self.value is None:
            self.value = entry
        else:
            self.value = (
                entry * (1 - self.smoothing_factor) + self.value * self.smoothing_factor
            )
        return self.value


class HistoryObservation(gym.ObservationWrapper):
   
    def __init__(self, env):
        super().__init__(env=env)
        self.observation_space = gym.spaces.Box(
            low=np.full(len(FLAGS.flags), 0, dtype=np.float32),
            high=np.full(len(FLAGS.flags), float("inf"), dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *args, **kwargs):
        self._steps_taken = 0
        self._state = np.zeros(
            (FLAGS.episode_len - 1, self.action_space.n), dtype=np.int32
        )
        obs=super().reset(*args, **kwargs)
        self.initial_ic = self.env.observation["IrInstructionCount"]
        # self.initial_rt = 0 if not len(self.env.observation["Runtime"])  == 0 else self.env.observation["Runtime"][0]
       
        if self.env.observation["Runtime"] is None:
            self.initial_rt = 0
        else:
            self.initial_rt = 0 if len(self.env.observation["Runtime"])  == 0 else self.env.observation["Runtime"][0]
        self.initial_auto = self.env.observation["Autophase"][51]
        
        return obs #super().reset(*args, **kwargs)

    def step(self, action: int):
        # print(f"Steps taken: {self._steps_taken}, Max steps: {FLAGS.episode_len}")
        assert self._steps_taken < FLAGS.episode_len, f"Exceeded max steps: {self._steps_taken} >= {FLAGS.episode_len}"

        if self._steps_taken < FLAGS.episode_len - 1:
            self._state[self._steps_taken][action] = 1
        self._steps_taken += 1
        

        # Call the parent step function to advance the environment
        obs, _, done, info = super().step(action)

        # Update after-action observations
        # self.after_rt =  0 if len(self.env.observation["Runtime"]) == 0 else self.env.observation["Runtime"][-1]
        
        if self.env.observation["Runtime"] is None:
            self.after_rt = 0
        else:
            self.after_rt = 0 if len(self.env.observation["Runtime"])  == 0 else self.env.observation["Runtime"][0]
        self.after_ic = self.env.observation["IrInstructionCount"]
        self.after_auto = self.env.observation["Autophase"][51]

        # Calculate the reward
        reward = self.calculate()

       
        # print(obs, reward, done, info)
        return obs, reward, done, info

        # return super().step(action)
    
    def calculate(self):
       
        #Runtime
        runtime = self.initial_rt - self.after_rt
        if runtime < 0:
            runtime = 0
        runtime *= 0.5
        
        #IR Instruction Cost
        ic = self.initial_ic - self.after_ic
        if ic < 0:
            ic = 0
        ic *= 0.3
       
        #Autophase Instruction
        auto = self.initial_auto - self.after_auto
        if auto < 0:
            auto = 0
        auto *= 0.2
        combined = runtime + (ic + auto) * 0.01
        
        return combined
        

    def observation(self, observation):
        return self._state


class Policy(nn.Module):
    """A very simple actor critic policy model."""

    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(
            (FLAGS.episode_len - 1) * len(FLAGS.flags), FLAGS.hidden_size
        )
        self.affine2 = nn.Linear(FLAGS.hidden_size, FLAGS.hidden_size)
        self.affine3 = nn.Linear(FLAGS.hidden_size, FLAGS.hidden_size)
        self.affine4 = nn.Linear(FLAGS.hidden_size, FLAGS.hidden_size)

        # Actor's layer
        self.action_head = nn.Linear(FLAGS.hidden_size, len(FLAGS.flags))

        # Critic's layer
        self.value_head = nn.Linear(FLAGS.hidden_size, 1)

        # Action & reward buffer
        self.saved_actions: List[SavedAction] = []
        self.rewards: List[float] = []

        
        self.moving_mean = MovingExponentialAverage(FLAGS.mean_smoothing)
        self.moving_std = MovingExponentialAverage(FLAGS.std_smoothing)

    def forward(self, x):
      
        x = F.relu(self.affine1(x))
        x = x.add(F.relu(self.affine2(x)))
        x = x.add(F.relu(self.affine3(x)))
        x = x.add(F.relu(self.affine4(x)))
       
        action_prob = F.softmax(self.action_head(x), dim=-1)
        
        state_values = self.value_head(x)
       
        return action_prob, state_values


def select_action(model, state, exploration_rate=0.0):
  
    state = torch.from_numpy(state.flatten()).float()
    probs, state_value = model(state)

    m = Categorical(probs)

    if random.random() < exploration_rate:
        action = torch.tensor(random.randrange(0, len(probs)))
    else:
        action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # The action to take.
    return action.item()


def TrainActorCritic(env):
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    max_ep_reward = -float("inf")
    avg_reward = MovingExponentialAverage(0.95)
    # avg_loss = MovingExponentialAverage(0.95)

    # for episode in range(1, FLAGS.episodes + 1):
    for episode in range(1, FLAGS.episodes_count + 1):   
        state = env.reset()
        ep_reward = 0
        sequence = []

        while True:
           
            action = select_action(model, state, FLAGS.exploration)
            sequence.append(action)            
            state, reward, done, _ = env.step(action)
            
            model.rewards.append(reward)

           
            ep_reward += reward
            if done:
                break

        
        if ep_reward  > max_ep_reward:
            
            best_sequence = sequence

        max_ep_reward = max(max_ep_reward, ep_reward)
        avg_reward.next(ep_reward)
        
        if (
            episode == 1
            or episode % FLAGS.log_interval == 0
            or episode == FLAGS.episodes_count
        ):
            print(
                f"Episode {episode}\t"
                f"Current reward: {ep_reward:.2f}\t"
                f"Avg reward: {avg_reward.value:.2f}\t"
                f"Best reward: {max_ep_reward:.2f}\t",
               
                flush=True,
            )
    action_names=[FLAGS.flags[int(idx)] for idx in best_sequence] 
    
    print(f"\nBest Sequence: {action_names} and Fitness: {max_ep_reward}")
    print(f"\nFinal performance (avg reward): {max_ep_reward}")
    return max_ep_reward, action_names


def make_env():
    FLAGS.env = "llvm-v0"
    
    env = env_from_flags(benchmark=benchmark_from_flags())
    env = ConstrainedCommandline(env, flags=FLAGS.flags)
    env = TimeLimit(env, max_episode_steps=FLAGS.episode_len)
    
    env = HistoryObservation(env)
    return env


def main(argv):
    """Main entry point."""
    del argv  # unused
    
    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    with make_env() as env:

        #without benchmark
        # env.reset()
    
        benchmarks = ["benchmark://cbench-v1/crc32","benchmark://cbench-v1/dijkstra","benchmark://cbench-v1/bzip2","benchmark://cbench-v1/jpeg-c"] #add additional
        #benchmarks = ["benchmark://chstone-v0/jpeg", "benchmark://chstone-v0/blowfish", "benchmark://chstone-v0/motion", "benchmark://chstone-v0/gsm"] #add additional
        for benchmark in benchmarks:
            print(f"Running Benchmark: {benchmark}")
            #if using benchmarks
            benchmark1 = benchmark
            env.reset(benchmark=benchmark1)

            print(f"Seed: {FLAGS.seed}")
            print(f"Episode length: {FLAGS.episode_len}")
            print(f"Number of episodes: {FLAGS.episodes_count}")
            print(f"Exploration: {FLAGS.exploration:.2%}")
            print(f"Observations: Runtime, IR Instruction Count, Autophase Instruction Count")
            print(f"Benchmark: {FLAGS.benchmark}")
            print(f"Action space: {env.action_space}\n")

            if FLAGS.iterations == 1:
                TrainActorCritic(env)
                return

            best_actions = []
            best_fitness=[]
            for i in range(1, FLAGS.iterations + 1):
                print(f"\n*** Iteration {i} of {FLAGS.iterations}")
                fitness, actions = TrainActorCritic(env)
                best_fitness.append(fitness)
                best_actions.append(actions)

            print("\n*** Reinforcement Learning Performance Review w/ Multiple Iterations")
            print(f"Algorthm Fitness: {best_fitness}\n")
            print(f"Best Fitness: {max(best_fitness)}")
            print(f"Avg Fitness: {statistics.mean(best_fitness)}")
            print(f"Worst Fitness: {min(best_fitness)}")
            print(f"Best Inviduals: {best_actions}")
            print("--------------------------------------")

if __name__ == "__main__":
    app.run(main)






