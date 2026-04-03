"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path
import heapq
from heapq import heappush, heappop
from itertools import count
import numpy as np
import time
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_move = None
        self.last_ghost_pos = None
        self.ghost_vector = (0, 0)
        self.name = "AStar_Interceptor"

    def step(self, map_state, my_pos, enemy_pos, step_number):
        # 1. Cập nhật hướng di chuyển của Ghost để đón đầu
        if enemy_pos is not None:
            if self.last_ghost_pos is not None:
                self.ghost_vector = (enemy_pos[0] - self.last_ghost_pos[0], 
                                     enemy_pos[1] - self.last_ghost_pos[1])
            self.last_ghost_pos = enemy_pos

        # 2. Dự đoán vị trí Ghost sẽ đến (Đón đầu 2 bước)
        target = enemy_pos
        if enemy_pos is not None and self.ghost_vector != (0, 0):
            # Dự đoán Ghost đi thêm 2 bước theo hướng cũ
            pred_r = enemy_pos[0] + self.ghost_vector[0] * 2
            pred_c = enemy_pos[1] + self.ghost_vector[1] * 2
            # Nếu vị trí dự đoán hợp lệ thì chọn làm mục tiêu
            if self._is_valid((pred_r, pred_c), map_state):
                target = (pred_r, pred_c)

        # 3. Tìm đường bằng A* (Tính toán dựa trên turns và physics)
        path = self._astar_turns(my_pos, target, map_state, self.last_move)
        
        if path:
            move, steps = path[0]
            self.last_move = move
            return (move, steps)
        
        # Fallback nếu A* lỗi
        return (Move.STAY, 1)

    def _astar_turns(self, start, goal, map_state, initial_move):
        def h(pos): return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # pq: (priority, g_cost, tie_breaker, current_pos, last_dir, path)
        tie_breaker = count()
        pq = [(h(start), 0, next(tie_breaker), start, initial_move, [])]
        visited = {} # (pos, last_dir) -> min_g

        while pq:
            f, g, _, curr, last_dir, path = heapq.heappop(pq)
            
            if abs(curr[0] - goal[0]) + abs(curr[1] - goal[1]) < 2:
                return path
            
            state = (curr, last_dir)
            if state in visited and visited[state] <= g: continue
            visited[state] = g
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                # Quy tắc: Nếu đi thẳng hướng cũ thì 2 bước, quẹo thì 1 bước
                is_straight = (last_dir is not None and move == last_dir)
                max_s = self.pacman_speed if is_straight else 1
                
                # Kiểm tra thực tế đi được mấy bước
                actual_s = 0
                temp_pos = curr
                for _ in range(max_s):
                    nxt = (temp_pos[0] + move.value[0], temp_pos[1] + move.value[1])
                    if not self._is_valid(nxt, map_state): break
                    actual_s += 1
                    temp_pos = nxt
                
                if actual_s > 0:
                    new_pos = temp_pos
                    new_g = g + 1
                    new_path = path + [(move, actual_s)]
                    heappush(pq, (new_g + h(new_pos)/self.pacman_speed, new_g, next(tie_breaker), new_pos, move, new_path))
        return []

    def _is_valid(self, pos, map_state):
        r, c = pos; h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    _is_valid_position = _is_valid

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart_BFS_Hider"

    def step(self, map_state, my_position, enemy_position, step_number) -> Move:
        # 1. Tính khoảng cách thực (số bước đi) từ Pacman đến mọi ô trên bản đồ
        pacman_dist_map = self._get_maze_distances(enemy_position, map_state)
        
        best_move = Move.STAY
        max_safety_score = -float('inf')

        # 2. Xét các hướng đi có thể của Ghost
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid_position(next_pos, map_state):
                # TÍNH ĐIỂM AN TOÀN CHO Ô NÀY
                # Khoảng cách thực tế từ Pacman đến ô này
                dist = pacman_dist_map[next_pos[0]][next_pos[1]]
                if dist == -1: dist = 100 # Ô mà Pacman không tới được
                
                # Số lối thoát từ ô này (Mobility)
                exits = self._count_exits(next_pos, map_state)
                
                # Né đường thẳng với Pacman (né Speed 2)
                line_penalty = 0
                if next_pos[0] == enemy_position[0] or next_pos[1] == enemy_position[1]:
                    line_penalty = -2 # Phạt nếu đứng thẳng hàng
                
                # CÔNG THỨC LƯỢNG GIÁ
                safety_score = dist * 2 + exits * 5 + line_penalty
                
                # Ngõ cụt là cực kỳ nguy hiểm
                if exits <= 1: safety_score -= 50 

                if safety_score > max_safety_score:
                    max_safety_score = safety_score
                    best_move = move

        return best_move

    def _get_maze_distances(self, start_pos, map_state):
        """BFS từ Pacman để tính số bước thực tế đến mọi ô."""
        h, w = map_state.shape
        dists = np.full((h, w), -1)
        queue = deque([start_pos])
        dists[start_pos[0]][start_pos[1]] = 0
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0 and dists[nr][nc] == -1:
                    dists[nr][nc] = dists[r][c] + 1
                    queue.append((nr, nc))
        return dists

    def _count_exits(self, pos, map_state):
        """Đếm số hướng đi được từ vị trí pos."""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            if self._is_valid_position((pos[0]+dr, pos[1]+dc), map_state):
                count += 1
        return count

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0
