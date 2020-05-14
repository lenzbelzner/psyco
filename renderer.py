from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pylab as plt
import numpy as np


class Renderer:
    def __init__(self, name, world):
        self.i = 0
        self.name = name
        self.world = world

    def make_frame(self, t):
        plt.gcf().clear()
        #plt.xlim(int(min(x)), int(max(x)))
        #plt.ylim(int(min(y)), int(max(y)))

        self.i += 1
        self.i = min(self.i, len(self.world.history))

        start = 0
        # start = max(0, self.i - int(len(self.world.history) * .2))

        for agent in self.world.agents:
            plt.scatter([x[0] for x in agent.history[start:self.i]], [x[1] for x in agent.history[start:self.i]], c=range(
                self.i - start), cmap='Blues', alpha=.5, s=2)

        plt.axis('off')
        plt.gca().set_aspect('equal')

        return mplfig_to_npimage(plt.gcf())

    def render_video(self):
        duration = int(len(self.world.history) * .1)
        animation = VideoClip(self.make_frame, duration=duration)
        # animation.write_gif(self.name, fps=10)
        animation.write_videofile(self.name + '.mp4', fps=10)


class ImgRenderer:
    def __init__(self, name, world):
        self.name = name
        self.world = world

    def render_img(self):
        plt.figure('trajectory')
        plt.gcf().clear()
        #plt.xlim(int(min(x)), int(max(x)))
        #plt.ylim(int(min(y)), int(max(y)))

        cmaps = ['winter', 'autumn']

        for i, agent in enumerate(self.world.agents):
            plt.scatter([x[0] for x in agent.history], [x[1] for x in agent.history], c=range(
                len(agent.history)), cmap=cmaps[i], s=4)

        # plt.axis('off')
        # plt.gca().set_aspect('equal')
        plt.savefig(self.name + '.png')

        # poincare plot
        plt.figure('poincare')
        plt.gcf().clear()

        for i, agent in enumerate(self.world.agents):
            plt.scatter(np.diff([x[0] for x in agent.history]), np.diff([x[1] for x in agent.history]), c=range(
                len(agent.history) - 1), cmap=cmaps[i], s=4)

        plt.savefig(self.name + '_poincare.png')


class MultiRenderer:
    def __init__(self, name, worlds):
        self.i = 0
        self.name = name
        self.worlds = worlds

    def make_frame(self, t):
        plt.gcf().clear()

        for world in self.worlds:
            self.i += 1
            self.i = min(self.i, len(world.history))
            start = 0
            for agent in world.agents:
                plt.scatter([x[0] for x in agent.history[start:self.i]], [x[1] for x in agent.history[start:self.i]], c=range(
                    self.i - start), cmap='Blues', alpha=.5, s=2)

        plt.axis('off')
        plt.gca().set_aspect('equal')

        return mplfig_to_npimage(plt.gcf())

    def render_video(self):
        duration = int(len(self.worlds[0].history) * .1)
        animation = VideoClip(self.make_frame, duration=duration)
        # animation.write_gif(self.name, fps=10)
        animation.write_videofile(self.name, fps=10)
