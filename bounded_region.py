import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Set up the grid
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

# Draw grid lines every 0.2
ticks = np.arange(0, 1.01, 0.2)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.7)

# Plot the boundary lines
x = np.linspace(0, 1, 500)
ax.plot(x, 0.85 - x, label='$c_1 + c_2 = 0.85$', color='blue')
ax.plot(x, x, label='$c_1 = c_2$', color='green')
ax.axhline(0.15, label='$c_2 = 0.15$', color='red')

# Define the polygon region bounded by the three lines
region = np.array([
    [0.15, 0.15],     # Intersection of y=0.15 and x=y
    [0.7, 0.15],      # Intersection of y=0.15 and x + y = 0.85
    [0.425, 0.425]    # Intersection of x=y and x + y = 0.85
])

# Add the shaded polygon
poly = Polygon(region, closed=True, color='purple', alpha=0.3)
ax.add_patch(poly)

# Mark and annotate points
points = [(0.3, 0.2), (0.4, 0.2), (0.4, 0.3), (0.4, 0.4), (0.5, 0.2), (0.5, 0.3)]
for i, (px, py) in enumerate(points, 1):
    ax.plot(px, py, 'ko')  # black dot
    ax.text(px + 0.015, py + 0.015, f'{i}', fontsize=12, color='black')

# Labels and legend
ax.set_xlabel('$c_1$')
ax.set_ylabel('$c_2$')
ax.set_title('Region bounded by constraints')
ax.legend(loc='upper right')
plt.tight_layout()

# Save the figure as PNG
plt.savefig("bounded_region.png", dpi=300)