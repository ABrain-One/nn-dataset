import os

nn_dir = os.path.join('ab', 'nn')
nn_module = str.replace(nn_dir, os.sep, '.')
default_config = ''
default_epochs = (1, 2, 5)
default_trials = 100
default_from_root = False
default_batch_power = 6
global from_root_g
