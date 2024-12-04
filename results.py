import matplotlib.pyplot as plt
import numpy as np

##################### CRC32

# Fitness scores across generations
generations = list(range(1, 17))
fitness_scores = [
    0.5913, 1.9550, 2.3150, 2.3150, 2.3301, 2.3305,
    2.3322, 2.3300, 2.3300, 2.3308, 2.3300, 2.3300,
    2.3308, 2.3306, 2.6200, 2.6200
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores, marker='o', linestyle='-', color='b')
plt.title("Genetic Algorithm Evolution - Cbench-v1/crc32")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_crc32.png")

# Show the plot
plt.show()

##################### BZIP2

generations = list(range(1, 11))
fitness_scores = [244.1900, 260.8650, 260.8650, 260.8650, 260.8650, 263.4500, 261.6950, 263.6700, 260.8650, 261.6950]

plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores, marker='o', linestyle='-', color='b')
plt.title("Genetic Algorithm Evolution - Cbench-v1/bzip2")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_bzip2.png")

# Show the plot
plt.show()

##################### Bar Chart comparing reinforcement learning and genetic algorithm results

# Data for the bar chart

# RL best fitness scores:
# 
# BZIP2 245.21500000000003
# CRC32 2.9949999999999997
# JPEG 634.1262025

# GA best fitness scores:
# 
# BZIP2 273.52
# CRC32 3.0563255000000003
# JPEG 704.3655325


# DO SEPARATE BAR CHARTS

# BZIP2
# GA: 273.52
# RL: 245.21500000000003

# CRC32
# GA: 3.0563255000000003
# RL: 2.9949999999999997

# JPEG
# GA: 704.3655325
# RL: 634.1262025

# Data for the bar chart
labels = ['BZIP2', 'CRC32', 'JPEG']
ga_scores = [273.52, 3.0563255000000003, 704.3655325]
rl_scores = [245.21500000000003, 2.9949999999999997, 634.1262025]

x = np.arange(len(labels))  # Convert range to NumPy array
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ga_scores, width, label='Genetic Algorithm')
rects2 = ax.bar(x + width/2, rl_scores, width, label='Reinforcement Learning')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_ylabel('Best Fitness Score')
ax.set_title('Genetic Algorithm vs Reinforcement Learning')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add labels above the bars
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

# Save the plot to the specified path
plt.savefig("data/bar_chart.png")

# Show the plot
plt.show()

# Data for CRC32 bar chart
labels_crc32 = ['Genetic Algorithm', 'Reinforcement Learning']
crc32_scores = [3.0563255000000003, 2.9949999999999997]

x_crc32 = np.arange(len(labels_crc32))
width = 0.5  # Adjusted for a single comparison

# Create the CRC32 bar chart
fig, ax = plt.subplots()
bars = ax.bar(x_crc32, crc32_scores, width, color=['tab:blue', 'tab:orange'])

# Add labels and title
ax.set_ylabel('Best Fitness Score')
ax.set_title('CRC32: Genetic Algorithm vs Reinforcement Learning')
ax.set_xticks(x_crc32)
ax.set_xticklabels(labels_crc32)

# Add labels above the bars
ax.bar_label(bars, padding=3)

# Save the plot to the specified path
plt.savefig("data/fitness_comparison_crc32.png")

# Show the plot
plt.show()
