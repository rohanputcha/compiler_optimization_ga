import compiler_gym

# Create the environment
env = compiler_gym.make("llvm-v0")

# List all observation spaces
print("Observation Spaces:")
for space in env.observation.spaces:
    print(f"  - {space}")

# List all reward spaces
print("\nReward Spaces:")
for space in env.reward.spaces:
    print(f"  - {space}")

# List all action spaces
print("\nAction Spaces:")
for action in env.action_space.names:
    print(f"  - {action}")



env.close()
