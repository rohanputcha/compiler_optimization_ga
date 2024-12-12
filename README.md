# Compiler Optimization Using Genetic Algorithms

This project explores compiler optimization by leveraging genetic algorithms (GA) and compares the performance with reinforcement learning (RL) techniques inspired by the CompilerGym framework. The goal is to improve loop vectorization and locality through automated compiler optimizations.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Running the Genetic Algorithm](#running-the-genetic-algorithm)
4. [Running the Reinforcement Learning Algorithm](#running-the-reinforcement-learning-algorithm)
5. [Tweaking Benchmarks](#tweaking-benchmarks)
6. [Project Structure](#project-structure)
7. [Dependencies](#dependencies)

---

## Getting Started

To get started, follow the instructions below to set up the environment and run the algorithms.

---

## Setting Up the Environment

1. **Create a virtual environment**:

   ```bash
   python -m venv env
   ```

2. **Activate the virtual environment**:

   - On macOS and Linux:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Genetic Algorithm

To run the genetic algorithm developed for this project, execute the following command:

```bash
python compiler_gym_ga.py
```

This script will run the GA-based optimization and output the results.

---

## Running the Reinforcement Learning Algorithm

To run the reinforcement learning algorithm based on the existing CompilerGym implementation, use the following command:

```bash
python compiler_gym_rl.py
```

This script will execute the RL-based optimization and provide a comparative baseline.

---

## Tweaking Benchmarks

You can modify the benchmarks used by each script directly in the respective code files. Look for sections in the code where benchmarks are defined or loaded.

### Example Code for `compiler_gym_ga.py`

In `compiler_gym_ga.py`, you might find code like this:

```python
from compiler_gym.envs import llvm

# Select a benchmark
env = llvm.make("llvm-v0")
benchmark = "cbench-v1/qsort"  # Example benchmark
env.reset(benchmark=benchmark)
```

You can change the `benchmark` variable to another supported benchmark from the CompilerGym suite. See our report for which benchmarks we have already run.

### Example Code for `compiler_gym_rl.py`

In `compiler_gym_rl.py`, the benchmark setup might look like:

```python
from compiler_gym.envs import llvm

# Load a specific benchmark
env = llvm.make("llvm-v0")
benchmark = "npb-v0/is"  # Example benchmark
env.reset(benchmark=benchmark)
```

Modify the `benchmark` string to use different benchmarks based on your needs.

---

## Project Structure

```
├── compiler_gym_ga.py       # Genetic Algorithm implementation
├── compiler_gym_rl.py       # Reinforcement Learning implementation
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── env/                     # Virtual environment (created after setup)
```

---

## Dependencies

Make sure the following dependencies are listed in your `requirements.txt` file. Example dependencies might include:

```plaintext
compiler_gym==0.2.4
numpy==1.21.2
matplotlib==3.4.3
torch==1.9.1
```

Install these using the following command:

```bash
pip install -r requirements.txt
```

---

## Notes

- Ensure you have **Python 3.8 or higher** installed.
- Refer to the [CompilerGym documentation](https://compiler.ai/) for more information on available benchmarks and environments.
- CompilerGym may randomly abort when using Mac OSX. Our code runs better on a Linux operating system.

---
