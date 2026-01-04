from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Example trajectory
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='blue')

# Add arrows along trajectory
for i in range(0, len(x)-1, 10):  # every 10th point
    arrow = FancyArrowPatch((x[i], y[i]), (x[i+1], y[i+1]),
                            arrowstyle='->', color='red', mutation_scale=15)
    plt.gca().add_patch(arrow)

plt.show()