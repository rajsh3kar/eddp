import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
class GaussMobilityModel:
    def __init__(self, grid_size, num_steps, sigma):
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.sigma = sigma
        self.user_position = None

    def initialize_user_position(self):
        x = random.uniform(0, self.grid_size)
        y = random.uniform(0, self.grid_size)
        self.user_position = np.array([x, y])

    def simulate_mobility(self):
        for _ in range(self.num_steps):
            dx = random.gauss(0, self.sigma)
            dy = random.gauss(0, self.sigma)
            new_x = self.user_position[0] + dx
            new_y = self.user_position[1] + dy

            if new_x < 0:
                new_x = -new_x
            elif new_x > self.grid_size:
                new_x = 2 * self.grid_size - new_x

            if new_y < 0:
                new_y = -new_y
            elif new_y > self.grid_size:
                new_y = 2 * self.grid_size - new_y

            self.user_position = (new_x, new_y)
            yield self.user_position

    def animate_mobility(self):
        fig, ax = plt.subplots()
        scatter = ax.scatter([], [])
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title('Gauss Mobility Model')

        def update_plot(frame):
            scatter.set_offsets(frame)
            return scatter,

        positions_generator = self.simulate_mobility()

        def animate(frame):
            position = next(positions_generator)
            scatter.set_offsets(position)
            return scatter,

        animation = FuncAnimation(fig, animate, frames=self.num_steps, interval=200, blit=True, repeat=False)
        plt.show()

# Parameters
grid_size = 2000
num_steps = 100000


'''# Create and simulate Gauss mobility model for a single user
mobility_model = GaussMobilityModel(grid_size, num_steps, sigma)
mobility_model.initialize_user_position()
mobility_model.animate_mobility()'''
