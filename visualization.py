
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

class ImageGenerator:
	def __init__(self, save_path=None):

		self.fig, self.ax = plt.subplots()
		self.save_path = save_path


	def __call__(self, rewards_history):

		moving_average = lambda x, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(**kw).mean().values
		plt.cla()
		self.ax.plot(rewards_history)
		self.ax.plot(moving_average(np.array(rewards_history),span=10), label='rewards ewma@10')

		if self.save_path:
			plt.savefig(self.save_path)

		plt.pause(1)
