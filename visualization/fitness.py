from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler


def plot_fitness(fitness: np.ndarray,
                 normalized: bool = False,
                 front_assignment: Optional[np.ndarray] = None) -> None:
    if normalized:
        scaler = MinMaxScaler(feature_range=(0, 1))
        fitness = scaler.fit_transform(fitness)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.get_cmap('viridis', front_assignment.max() - front_assignment.min() + 1)
    bounds = range(front_assignment.min(), front_assignment.max() + 2)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    xs = fitness[:, 0]
    ys = fitness[:, 1]
    zs = fitness[:, 2]

    s3d = ax.scatter(xs=xs, ys=ys, zs=zs, c=front_assignment, cmap=cmap, norm=norm)
    ax.set_xlabel('Edge Value')
    ax.set_ylabel('Connectivity')
    ax.set_zlabel('Deviation')
    cb = fig.colorbar(s3d, ax=ax, ticks=front_assignment+0.5)
    cb.set_ticklabels(front_assignment)

    plt.show()


def visualize_fitness_history(fitness_path: str):
    df = pd.read_csv(fitness_path, dtype={'generation': int, 'front': int})

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.get_cmap('viridis', df['front'].max() - df['front'].min() + 1)
    bounds = range(df['front'].min(), df['front'].max() + 2)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    s3d = ax.scatter([], [], [], c=[], cmap=cmap, norm=norm)

    cb = fig.colorbar(s3d, ax=ax, ticks=np.arange(df['front'].min(), df['front'].max()+1) + 0.5)
    cb.set_ticklabels(np.arange(df['front'].min(), df['front'].max()+1))

    def animate(generation):
        ax.set_title(f'Generation {generation}')
        data = df[df['generation'] == generation]

        s3d._offsets3d = (data['edge value'].values, data['connectivity'].values, data['deviation'].values)
        s3d.set_array(data['front'])

    ax.set_xlabel('Edge Value')
    ax.set_ylabel('Connectivity')
    ax.set_zlabel('Deviation')
    ax.set_xlim(df['edge value'].min(), df['edge value'].max())
    ax.set_ylim(df['connectivity'].min(), df['connectivity'].max())
    ax.set_zlim(df['deviation'].min(), df['deviation'].max())

    ani = FuncAnimation(fig, animate, frames=df['generation'].max())
    plt.show()