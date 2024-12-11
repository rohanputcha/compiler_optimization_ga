import compiler_gym

# Create the environment
env = compiler_gym.make("llvm-v0")
count =0
# List all observation spaces
print("Observation Spaces:")
for space in env.observation.spaces:
    count = count + 1
    print(f"  - {space}")
print(f'Total number of observation spaces: {count}')

# List all reward spaces
count = 0
print("\nReward Spaces:")
for space in env.reward.spaces:
    count = count + 1    
    print(f"  - {space}")
print(f'Total number of reward spaces: {count}')

# List all action spaces
print("\nAction Spaces:")
count =0
for action in env.action_space.names:
    count = count + 1
    print(f"  - {action}")
print(f'Total number of action spaces: {count}')

env.close()
