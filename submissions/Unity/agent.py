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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

# ADDED: deque cho BFS, typing cho type hints
from collections import deque
from typing import Dict, List, Tuple


# ===========================================================================
# ADDED: Hằng số và helper functions dùng chung cho cả 2 agent
# ===========================================================================

# ADDED: Danh sách các hướng di chuyển kèm delta (row, col)
MOVE_DELTAS: List[Tuple[Move, int, int]] = [
    (Move.UP,    -1,  0),
    (Move.DOWN,   1,  0),
    (Move.LEFT,   0, -1),
    (Move.RIGHT,  0,  1),
]

INF = float('inf')


def _bfs_dist_map(start: Tuple[int, int],
                  map_state: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    ADDED: BFS từ `start`, trả về dict {cell: khoảng_cách_BFS}.
    Dùng để tính khoảng cách thực tế trong maze (không phải Manhattan).
    """
    dist: Dict[Tuple[int, int], int] = {start: 0}
    queue: deque = deque([start])
    while queue:
        r, c = queue.popleft()
        d = dist[(r, c)]
        for _, dr, dc in MOVE_DELTAS:
            nxt = (r + dr, c + dc)
            if nxt not in dist and _is_walkable(map_state, nxt):
                dist[nxt] = d + 1
                queue.append(nxt)
    return dist


def _is_walkable(map_state: np.ndarray, pos: Tuple[int, int]) -> bool:
    """
    ADDED: Kiểm tra ô có thể đi được không (value == 0, trong bounds).
    Tách riêng để dùng chung giữa các helper function module-level.
    """
    r, c = pos
    h, w = map_state.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return int(map_state[r, c]) == 0


def _predict_ghost_next(ghost_pos: Tuple[int, int],
                        pac_pos: Tuple[int, int],
                        map_state: np.ndarray) -> Tuple[int, int]:
    """
    ADDED: Dự đoán vị trí tiếp theo của Ghost nếu nó chạy tối ưu.
    Ghost chọn ô lân cận nào có BFS distance đến pac_pos lớn nhất.
    Trả về ghost_pos hiện tại nếu không có ô nào tốt hơn.
    """
    pac_dist = _bfs_dist_map(pac_pos, map_state)
    best_pos = ghost_pos
    best_dist = pac_dist.get(ghost_pos, 0)

    gr, gc = ghost_pos
    for _, dr, dc in MOVE_DELTAS:
        nxt = (gr + dr, gc + dc)
        if _is_walkable(map_state, nxt):
            d = pac_dist.get(nxt, 0)
            if d > best_dist:
                best_dist = d
                best_pos = nxt
    return best_pos


def _get_pacman_actions(pos: Tuple[int, int],
                        map_state: np.ndarray,
                        pacman_speed: int
                        ) -> List[Tuple[Move, int, Tuple[int, int]]]:
    """
    ADDED: Liệt kê tất cả hành động hợp lệ của Pacman dưới dạng
    (move, steps, landing_pos).
    - steps=1: ô kề walkable
    - steps=2: cả ô kề lẫn ô tiếp theo đều walkable, và pacman_speed >= 2
    """
    actions: List[Tuple[Move, int, Tuple[int, int]]] = []
    r, c = pos
    for move, dr, dc in MOVE_DELTAS:
        step1 = (r + dr, c + dc)
        if not _is_walkable(map_state, step1):
            continue
        actions.append((move, 1, step1))
        if pacman_speed >= 2:
            step2 = (r + 2 * dr, c + 2 * dc)
            if _is_walkable(map_state, step2):
                actions.append((move, 2, step2))
    return actions


# ===========================================================================
# PacmanAgent
# ===========================================================================

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Thuật toán: BFS Pursuit + Predictive Interception + Speed Exploitation
    - BFS thay vì Manhattan để tìm đường đúng trong maze
    - Dự đoán Ghost sẽ chạy đâu → chặn trước thay vì đuổi theo
    - Tận dụng steps=2 để di chuyển nhanh gấp đôi Ghost
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "BFS Intercept Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            (Move, steps): Direction và số bước (1 hoặc 2)
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            return self._explore(my_position, map_state)

        # ADDED: Dự đoán vị trí Ghost sau 1 bước (giả sử Ghost chạy tối ưu)
        predicted_target = _predict_ghost_next(target, my_position, map_state)

        # ADDED: Tính BFS distance map từ predicted_target đến mọi ô
        target_dist = _bfs_dist_map(predicted_target, map_state)

        # ADDED: Liệt kê tất cả actions hợp lệ (steps=1 và steps=2)
        actions = _get_pacman_actions(my_position, map_state, self.pacman_speed)
        if not actions:
            return (Move.STAY, 1)

        # ADDED: Chọn action tốt nhất dựa trên BFS distance đến predicted_target
        # Score = (dist_đến_ghost, -steps): dist nhỏ hơn tốt hơn, steps nhiều hơn phá hoà
        best_action = None
        best_score = (INF, 0)

        for move, steps, landing in actions:
            dist = target_dist.get(landing, INF)
            score = (dist, -steps)
            if score < best_score:
                best_score = score
                best_action = (move, steps)

        return best_action if best_action else (Move.STAY, 1)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """
        ADDED: Fallback khi không biết enemy ở đâu.
        Ưu tiên steps=2 để di chuyển nhanh nhất có thể.
        """
        actions = _get_pacman_actions(my_position, map_state, self.pacman_speed)
        if actions:
            actions.sort(key=lambda a: -a[1])   # ưu tiên steps lớn hơn
            move, steps, _ = actions[0]
            return (move, steps)
        return (Move.STAY, 1)

    # -- Giữ lại từ template (dùng cho _explore fallback nếu cần) -----------

    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move,
                         map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        return self._max_valid_steps(pos, move, map_state, 1) == 1

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0


# ===========================================================================
# GhostAgent
# ===========================================================================

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught

    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        # Lưu vị trí trước đó để tránh quay đầu ngay lập tức
        self.last_position = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Use current sighting, fallback to last known
        threat = enemy_position or self.last_known_enemy_pos

        # Lấy tất cả neighbor hợp lệ theo thứ tự cố định để luôn deterministic
        neighbors = self._get_valid_neighbors(my_position, map_state)
        if not neighbors:
            return Move.STAY

        best_move = None
        best_key = None

        if threat is None:
            # Không biết Pacman ở đâu: ưu tiên ô có nhiều lối thoát hơn
            for order_idx, (move, nxt) in enumerate(neighbors):
                exits = self._count_exits(nxt, map_state)
                avoid_backtrack = 1 if self.last_position is None or nxt != self.last_position else 0
                key = (exits, avoid_backtrack, -order_idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_move = move
        else:
            # Biết (hoặc nhớ) Pacman: tối đa BFS distance, rồi số lối thoát, rồi tránh backtrack
            pac_dist = _bfs_dist_map(threat, map_state)
            for order_idx, (move, nxt) in enumerate(neighbors):
                dist = pac_dist.get(nxt, -1)
                exits = self._count_exits(nxt, map_state)
                avoid_backtrack = 1 if self.last_position is None or nxt != self.last_position else 0
                key = (dist, exits, avoid_backtrack, -order_idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_move = move

        # Cập nhật memory vị trí trước khi trả về move
        if best_move is None:
            return Move.STAY
        self.last_position = my_position
        return best_move

    def _get_valid_neighbors(self, pos: tuple, map_state: np.ndarray):
        """Trả về danh sách (move, next_pos) hợp lệ theo thứ tự cố định."""
        r, c = pos
        neighbors = []
        for move, dr, dc in MOVE_DELTAS:
            nxt = (r + dr, c + dc)
            if _is_walkable(map_state, nxt):
                neighbors.append((move, nxt))
        return neighbors

    def _count_exits(self, pos: tuple, map_state: np.ndarray) -> int:
        """Đếm số ô hàng xóm đi được của một vị trí."""
        r, c = pos
        exits = 0
        for _, dr, dc in MOVE_DELTAS:
            if _is_walkable(map_state, (r + dr, c + dc)):
                exits += 1
        return exits

    # -- Giữ lại từ template -------------------------------------------------

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0
