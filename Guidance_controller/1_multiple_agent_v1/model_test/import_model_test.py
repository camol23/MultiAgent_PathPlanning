import tensorflow as tf
import numpy as np


import actor_tg4_tf
model = actor_tg4_tf.load_model()

#
# tg4 Agent doesn't consider the obstacle
#
# Observations = [Xe Ye Th_e Th Vx Vy W guidline]
#  