import sys
sys.path.insert(0, '../../src')
import importlib.util, numpy as np

spec = importlib.util.spec_from_file_location('agent', 'agent.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from environment import Move
ghost = m.GhostAgent(pacman_speed=2)
map_state = np.zeros((5,5), dtype=int)
move = ghost.step(map_state, (2,2), (4,4), 1)
print('Ghost move:', move)
print('IMPORT OK')
