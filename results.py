import matplotlib.pyplot as plt
import numpy as np

##################### CRC32

# Fitness scores across generations
generations = list(range(1, 16))
fitness_scores_1 = [
    0.5913, 1.9550, 2.3150, 2.3150, 2.3301, 2.3305,
    2.3322, 2.3300, 2.3300, 2.3308, 2.3300, 2.3300,
    2.3308, 2.3306, 2.6200
]
fitness_scores_2 = [
    2.9053, 2.9051, 2.9055, 2.9501, 2.9350, 3.0302, 2.9507, 2.9505, 2.9062, 2.9550, 2.9510, 2.9357, 3.0250, 3.0250, 3.0250
]
fitness_scores_3 = [
    0.9502, 0.9514, 0.9514, 0.9511, 1.0512, 1.0529, 1.9900, 1.9902, 1.0069, 0.9508, 1.3750, 0.9501, 0.9517, 0.9504, 1.7300
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores_1, marker='o', linestyle='-', color='r')
plt.plot(generations, fitness_scores_2, marker='o', linestyle='-', color='b')
plt.plot(generations, fitness_scores_3, marker='o', linestyle='-', color='g')
plt.title("Genetic Algorithm Evolution - Cbench-v1/crc32")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_crc32.png")

# Show the plot
plt.show()

##################### BZIP2
# Fitness scores across generations
generations = list(range(1, 16))
fitness_scores_1 = [
    174.9950, 177.0050, 176.4150, 273.5200, 246.5600, 174.9950,
    171.7300, 174.9950, 265.1200, 264.4500, 265.1200, 269.8200,
    264.4500, 192.4358, 195.4416
]
fitness_scores_2 = [
    104.6807, 104.6815, 109.0853, 156.8054, 151.1153, 151.4503, 
    152.4006, 151.4510, 151.4632, 151.4600, 105.2704, 105.2700, 
    105.2708, 105.5261, 105.5270
]
fitness_scores_3 = [
    244.1900, 260.8650, 260.8650, 260.8650, 260.8650, 263.4500, 
    261.6950, 263.6700, 260.8650, 261.6950, 261.6950, 260.8650, 
    263.7900, 260.8650, 259.5700
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores_1, marker='o', linestyle='-', color='r')
plt.plot(generations, fitness_scores_2, marker='o', linestyle='-', color='b')
plt.plot(generations, fitness_scores_3, marker='o', linestyle='-', color='g')
plt.title("Genetic Algorithm Evolution - Cbench-v1/bzip2")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_bzip2.png")

# Show the plot
plt.show()

##################### JPEG
# Fitness scores across generations
generations = list(range(1, 16))
fitness_scores_1 = [
    306.5029, 519.1751, 521.7257, 521.7255, 503.2728, 521.7281,
    518.3407, 503.2742, 503.8305, 505.3877, 505.3853, 503.9812,
    503.8313, 503.9805, 503.9826
]
fitness_scores_2 = [
    652.4190, 653.0750, 653.0765, 653.0759, 653.8669, 662.3803, 
    662.3805, 704.3553, 704.3556, 704.3655, 694.5561, 694.5583, 
    694.5550, 694.5568, 694.5560
]
fitness_scores_3 = [
    685.5100, 685.5106, 689.0809, 685.5115, 689.0813, 689.0805, 
    698.0761, 689.0808, 689.0800, 633.6851, 633.6870, 633.6872, 
    634.4728, 691.1076, 633.9738
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores_1, marker='o', linestyle='-', color='r')
plt.plot(generations, fitness_scores_2, marker='o', linestyle='-', color='b')
plt.plot(generations, fitness_scores_3, marker='o', linestyle='-', color='g')
plt.title("Genetic Algorithm Evolution - Cbench-v1/jpeg-d")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_jpeg.png")

# Show the plot
plt.show()

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


##################### CRC32

# Fitness scores across generations
generations = [1, 10, 20, 30, 40, 50, 60]
fitness_scores_1 = [
    0.57, 2.81, 2.81, 2.95, 2.95, 2.99, 2.99
]
fitness_scores_2 = [
    0.00, 0.05, 2.91, 2.91, 2.95, 2.95, 2.95
]
fitness_scores_3 = [
    0.08, 0.08, 0.97, 2.27, 2.27, 2.27, 2.91
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores_1, marker='o', linestyle='-', color='r')
plt.plot(generations, fitness_scores_2, marker='o', linestyle='-', color='b')
plt.plot(generations, fitness_scores_3, marker='o', linestyle='-', color='g')
plt.title("Reinforcement Learning - Cbench-v1/crc32")
plt.xlabel("Episode")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_crc32_rl.png")

# Show the plot
plt.show()

##################### BZIP2

# Fitness scores across generations
generations = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
fitness_scores_1 = [
    0.00, 182.79, 182.79, 182.79, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1, 240.1,
]
fitness_scores_2 = [
    0.00, 65.75, 65.75, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36, 197.36
]
fitness_scores_3 = [
    148.53, 148.53, 148.53, 148.53, 170.04, 174.04, 174.04, 174.04, 174.04, 174.04, 245.22, 245.22, 245.22, 245.22, 245.22, 245.22
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores_1, marker='o', linestyle='-', color='r')
plt.plot(generations, fitness_scores_2, marker='o', linestyle='-', color='b')
plt.plot(generations, fitness_scores_3, marker='o', linestyle='-', color='g')
plt.title("Reinforcement Learning - Cbench-v1/bzip2")
plt.xlabel("Episode")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_bzip2_rl.png")

# Show the plot
plt.show()


##################### JPEG

# Fitness scores across generations
generations = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
fitness_scores_1 = [
    0.00, 383.96, 383.96, 383.96, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31, 628.31
]
fitness_scores_2 = [
    0.00, 111.91, 111.91, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10, 506.10
]
fitness_scores_3 = [
    380.15, 380.15, 380.15, 380.15, 380.15, 380.15, 380.15, 380.15, 380.15, 380.15,634.13, 634.13, 634.13, 634.13, 634.13, 634.13
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_scores_1, marker='o', linestyle='-', color='r')
plt.plot(generations, fitness_scores_2, marker='o', linestyle='-', color='b')
plt.plot(generations, fitness_scores_3, marker='o', linestyle='-', color='g')
plt.title("Reinforcement Learning - Cbench-v1/jpeg")
plt.xlabel("Episode")
plt.ylabel("Best Fitness Score")
plt.grid(True)

# Save the plot to the specified path
plt.savefig("data/fitness_evolution_cbench_jpeg_rl.png")

# Show the plot
plt.show()
