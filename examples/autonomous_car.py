#!/usr/bin/env python3

"""
Autonomous exploration example for Trilobot.

The robot spends most of its time driving from a forward-facing distance check,
but periodically sweeps the ultrasonic sensor across several headings to find a
better route around obstacles. A short turn-memory term biases decisions away
from repeatedly choosing the same side, which helps it cover more of a room.

Stop the example by pressing button A.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


Colour = Tuple[int, int, int]


@dataclass(frozen=True)
class AutonomousCarConfig:
    scan_angles: Tuple[int, ...] = (-80, -45, 0, 45, 80)
    scan_pause_s: float = 0.05
    loop_delay_s: float = 0.06
    stale_scan_s: float = 0.90
    proactive_scan_s: float = 2.40
    danger_distance: float = 14.0
    caution_distance: float = 28.0
    cruise_distance: float = 55.0
    max_distance: float = 140.0
    cruise_speed: float = 0.62
    cautious_speed: float = 0.42
    steer_gain: float = 0.48
    escape_reverse_speed: float = 0.45
    escape_turn_speed: float = 0.74
    escape_reverse_s: float = 0.22
    escape_turn_s: float = 0.32
    front_history_size: int = 5
    recent_turn_memory: int = 8
    exploration_bonus: float = 10.0
    corridor_support_weight: float = 0.55
    isolation_penalty: float = 0.75
    forward_bias_gain: float = 0.55
    target_heading_bias: float = 16.0
    target_heading_hold_s: float = 1.1
    narrow_gap_penalty: float = 18.0
    escape_angle_bias: float = 0.45
    edge_penalty: float = 18.0
    distance_advantage_gain: float = 0.70
    front_timeout_ms: int = 25
    front_samples: int = 1
    scan_timeout_ms: int = 35
    scan_samples: int = 2
    stuck_window_s: float = 3.0
    stuck_escape_count: int = 4
    stuck_spin_s: float = 0.55
    open_space_distance: float = 70.0
    open_space_speed: float = 0.82
    speed_accel_rate: float = 0.10   # max speed increase per loop cycle
    speed_decel_rate: float = 0.18   # max speed decrease per loop cycle
    open_space_scan_s: float = 1.2   # proactive scan interval when all angles are clear
    history_alpha: float = 0.25     # EMA smoothing: higher = more reactive to recent scans
    history_bias_gain: float = 8.0  # score bonus/penalty scale for historical openness
    side_correction_gain: float = 0.10  # proportional steer correction away from closer side wall
    gap_interpolation: bool = True      # create virtual headings between adjacent clear scan angles
    gap_score_scale: float = 0.85       # distance scale factor for interpolated gap headings
    heading_momentum_gain: float = 4.0  # score bonus for staying on current heading (reduces zigzag)
    approach_rate_window: int = 4       # number of front samples for approach rate calculation
    approach_rate_decel_gain: float = 0.30  # extra deceleration per cm/s approach rate
    heading_smoothing_alpha: float = 0.35  # low-pass filter on heading (higher = more responsive)
    wall_following_gain: float = 0.12     # proportional gain for wall-following in corridors
    wall_following_threshold: float = 45.0  # max distance to both sides to trigger wall-following
    exploration_decay: float = 0.92       # decay factor for exploration memory per loop cycle
    exploration_penalty_scale: float = 12.0  # score penalty scale for recently chosen angles
    per_angle_rate_window: int = 3        # samples for per-angle approach rate tracking
    spatial_memory_cells: int = 5          # number of angular sectors in spatial memory
    spatial_memory_decay: float = 0.97     # decay rate for spatial memory per loop cycle
    spatial_obstacle_threshold: float = 25.0  # distance below which a cell is marked as blocked
    frontier_exploration_gain: float = 6.0  # score bonus for under-explored directions
    frontier_decay: float = 0.95           # decay rate for frontier memory per loop cycle
    dead_end_front_threshold: float = 35.0  # max front distance to trigger dead end check
    dead_end_side_threshold: float = 40.0   # max side distance to confirm dead end
    dead_end_recovery_turn_s: float = 0.45  # duration of dead end recovery turn
    escape_escalation_window: float = 5.0   # time window for escape escalation
    escape_escalation_levels: int = 3       # number of escalation levels
    escape_level_duration: float = 1.5      # seconds per escalation level
    path_commitment_s: float = 0.8          # minimum time to commit to a chosen heading
    path_commitment_bonus: float = 8.0      # score bonus for maintaining current heading
    predictive_horizon_s: float = 0.5       # seconds ahead to predict collisions
    predictive_collision_margin: float = 8.0  # extra cm margin for predicted collisions
    boredom_threshold_s: float = 8.0        # seconds before boredom triggers exploration
    boredom_exploration_bonus: float = 12.0  # score bonus when bored to force new direction
    environment_complexity_window: int = 10  # samples for environment complexity estimation
    complexity_scan_multiplier: float = 0.6   # scan interval multiplier in complex environments
    adaptive_scan_base: float = 2.4          # baseline proactive scan interval
    proximity_urgency_gain: float = 0.8      # additional deceleration when very close to obstacles
    proximity_urgency_distance: float = 20.0  # distance threshold for urgency mode
    oscillation_window: int = 6               # number of recent heading changes to detect oscillation
    oscillation_threshold: int = 4            # sign changes in window to trigger oscillation break
    oscillation_break_turn: float = 0.6       # forced turn duration when oscillation detected
    gap_centering_gain: float = 0.15          # proportional gain to center car in detected gaps
    gap_entry_distance: float = 35.0          # max side distance to consider entering a gap
    gap_entry_slowdown: float = 0.25          # speed reduction factor when entering narrow gaps
    scan_confidence_window: int = 5           # recent scans to evaluate confidence
    scan_confidence_threshold: float = 0.3    # variance threshold for low-confidence scan
    low_confidence_scan_s: float = 0.6        # reduced scan interval when confidence is low


@dataclass(frozen=True)
class MotionCommand:
    mode: str
    left_speed: float
    right_speed: float
    heading: int
    colour: Colour


@dataclass
class AutonomousCarController:
    config: AutonomousCarConfig
    front_history: Deque[float] = field(init=False)
    recent_turns: Deque[int] = field(init=False)
    stuck_escape_times: Deque[float] = field(init=False)
    last_scan: Dict[int, float] = field(default_factory=dict)
    angle_ema: Dict[int, float] = field(default_factory=dict)
    last_scan_time: float = field(default=-math.inf)
    target_heading: int = field(default=0)
    target_heading_time: float = field(default=-math.inf)
    current_speed: float = field(default=0.0)
    current_heading: int = field(default=0)
    front_rate_history: Deque[Tuple[float, float]] = field(init=False)
    smoothed_heading: float = field(default=0.0)
    exploration_memory: Dict[int, float] = field(default_factory=dict)
    angle_rate_history: Dict[int, Deque[Tuple[float, float]]] = field(default_factory=dict)
    spatial_memory: Dict[int, float] = field(default_factory=dict)
    frontier_map: Dict[int, float] = field(default_factory=dict)
    escape_escalation_level: int = field(default=0)
    last_escape_time: float = field(default=-math.inf)
    path_commitment_time: float = field(default=-math.inf)
    path_commitment_heading: int = field(default=0)
    dead_end_recovery_time: float = field(default=-math.inf)
    total_distance_traveled: float = field(default=0.0)
    last_speed: float = field(default=0.0)
    heading_initialized: bool = field(default=False)
    boredom_timer: float = field(default=0.0)
    last_heading_change_time: float = field(default=0.0)
    complexity_history: Deque[float] = field(init=False)
    last_heading_for_boredom: int = field(default=0)
    consecutive_small_turns: int = field(default=0)
    heading_history: Deque[int] = field(init=False)
    last_plan_time: float = field(default=0.0)
    scan_confidence_history: Deque[float] = field(init=False)

    def __post_init__(self):
        self.front_history = deque(maxlen=self.config.front_history_size)
        self.recent_turns = deque(maxlen=self.config.recent_turn_memory)
        self.stuck_escape_times = deque(maxlen=self.config.stuck_escape_count + 1)
        self.front_rate_history = deque(maxlen=self.config.approach_rate_window)
        self.angle_rate_history = {}
        self.complexity_history = deque(maxlen=self.config.environment_complexity_window)
        self.heading_history = deque(maxlen=self.config.oscillation_window)
        self.scan_confidence_history = deque(maxlen=self.config.scan_confidence_window)
        for i in range(self.config.spatial_memory_cells):
            self.spatial_memory[i] = 0.0
            self.frontier_map[i] = 1.0

    def is_stuck(self, now: float) -> bool:
        cutoff = now - self.config.stuck_window_s
        while self.stuck_escape_times and self.stuck_escape_times[0] < cutoff:
            self.stuck_escape_times.popleft()
        return len(self.stuck_escape_times) >= self.config.stuck_escape_count

    def sanitize_distance(self, distance: float) -> float:
        if not math.isfinite(distance) or distance <= 0.0:
            return 0.0
        return min(distance, self.config.max_distance)

    def observe_front(self, front_distance: float) -> float:
        front_distance = self.sanitize_distance(front_distance)
        if front_distance > 0.0:
            self.front_history.append(front_distance)
        if not self.front_history:
            return front_distance
        ordered = sorted(self.front_history)
        return ordered[len(ordered) // 2]

    def track_approach_rate(self, front_distance: float, now: float) -> float:
        self.front_rate_history.append((now, front_distance))
        if len(self.front_rate_history) < 2:
            return 0.0
        entries = list(self.front_rate_history)
        dt = entries[-1][0] - entries[0][0]
        if dt <= 0.0:
            return 0.0
        # positive = approaching (distance decreasing), negative = receding
        return (entries[0][1] - entries[-1][1]) / dt

    def update_angle_rates(self, scan: Dict[int, float], now: float):
        for angle, dist in scan.items():
            if dist <= 0.0:
                continue
            if angle not in self.angle_rate_history:
                self.angle_rate_history[angle] = deque(maxlen=self.config.per_angle_rate_window)
            self.angle_rate_history[angle].append((now, dist))

    def get_angle_approach_rate(self, angle: int) -> float:
        hist = self.angle_rate_history.get(angle)
        if not hist or len(hist) < 2:
            return 0.0
        entries = list(hist)
        dt = entries[-1][0] - entries[0][0]
        if dt <= 0.0:
            return 0.0
        return (entries[0][1] - entries[-1][1]) / dt

    def note_turn(self, heading: int):
        if heading < 0:
            self.recent_turns.append(-1)
        elif heading > 0:
            self.recent_turns.append(1)

    def should_scan(self, front_distance: float, now: float) -> bool:
        front_distance = self.observe_front(front_distance)
        if front_distance <= 0.0:
            return True
        if not self.last_scan:
            return True
        if front_distance <= self.config.caution_distance:
            return True
        if self.is_low_confidence() and now - self.last_scan_time >= self.config.low_confidence_scan_s:
            return True
        positive_scan = [v for v in self.last_scan.values() if v > 0.0]
        was_open = bool(positive_scan) and all(v > self.config.open_space_distance for v in positive_scan)
        proactive_interval = self.config.open_space_scan_s if was_open else self.get_adaptive_scan_interval()
        if now - self.last_scan_time >= proactive_interval:
            return True
        if now - self.last_scan_time >= self.config.stale_scan_s:
            left = self.sanitize_distance(self.last_scan.get(-45, 0.0))
            right = self.sanitize_distance(self.last_scan.get(45, 0.0))
            if abs(left - right) >= 18.0:
                return True
        return False

    def update_scan(self, scan: Dict[int, float], now: float):
        self.last_scan = {angle: self.sanitize_distance(distance) for angle, distance in scan.items()}
        self.last_scan_time = now
        front_distance = self.last_scan.get(0, 0.0)
        if front_distance > 0.0:
            self.front_history.append(front_distance)
        alpha = self.config.history_alpha
        for angle, dist in self.last_scan.items():
            if dist > 0.0:
                if angle not in self.angle_ema:
                    self.angle_ema[angle] = dist
                else:
                    self.angle_ema[angle] = alpha * dist + (1.0 - alpha) * self.angle_ema[angle]
        self.update_angle_rates(self.last_scan, now)
        self.update_spatial_memory(self.last_scan)
        self.update_frontier_map(self.last_scan)
        self.update_complexity(self.last_scan)
        self.update_scan_confidence(self.last_scan)

    def corridor_support(self, ordered_scan: Tuple[Tuple[int, float], ...], index: int) -> Tuple[float, int]:
        neighbours = []
        if index > 0:
            neighbours.append(ordered_scan[index - 1][1])
        if index + 1 < len(ordered_scan):
            neighbours.append(ordered_scan[index + 1][1])
        if not neighbours:
            return ordered_scan[index][1], 0
        if len(neighbours) == 1:
            distance = ordered_scan[index][1]
            # Edge headings only have one adjacent reading, so treat the open side as uncertain.
            return (neighbours[0] + distance * 0.35) / 2.0, 1
        return sum(neighbours) / len(neighbours), len(neighbours)

    def build_gap_headings(self, ordered_scan: Tuple[Tuple[int, float], ...]) -> Tuple[Tuple[int, float], ...]:
        if not self.config.gap_interpolation or len(ordered_scan) < 2:
            return ordered_scan
        gaps = []
        for i in range(len(ordered_scan) - 1):
            a1, d1 = ordered_scan[i]
            a2, d2 = ordered_scan[i + 1]
            if d1 > 0.0 and d2 > 0.0:
                mid_angle = (a1 + a2) // 2
                mid_dist = min(d1, d2) * self.config.gap_score_scale
                gaps.append((mid_angle, mid_dist))
        combined = list(ordered_scan) + gaps
        combined.sort(key=lambda x: x[0])
        return tuple(combined)

    def should_use_gap_headings(self, front_distance: float) -> bool:
        return self.config.caution_distance <= front_distance < self.config.cruise_distance

    def gap_bonus_for_angle(self, angle: int, ordered_scan: Tuple[Tuple[int, float], ...]) -> float:
        if not self.config.gap_interpolation or len(ordered_scan) < 2:
            return 0.0
        for i in range(len(ordered_scan) - 1):
            a1, d1 = ordered_scan[i]
            a2, d2 = ordered_scan[i + 1]
            if d1 > 0.0 and d2 > 0.0 and a1 < angle < a2:
                mid_dist = min(d1, d2) * self.config.gap_score_scale
                return mid_dist * 0.15
        return 0.0

    def heading_memory_bonus(self, angle: int, now: float | None) -> float:
        if now is None or self.target_heading == 0:
            return 0.0
        age = now - self.target_heading_time
        if age < 0.0 or age > self.config.target_heading_hold_s:
            return 0.0

        age_ratio = 1.0 - (age / self.config.target_heading_hold_s)
        angle_gap = abs(angle - self.target_heading)
        angle_ratio = max(0.0, 1.0 - angle_gap / 90.0)
        return self.config.target_heading_bias * age_ratio * angle_ratio

    def exploration_penalty(self, angle: int) -> float:
        decay = self.config.exploration_decay
        for a in list(self.exploration_memory.keys()):
            self.exploration_memory[a] *= decay
            if self.exploration_memory[a] < 0.05:
                del self.exploration_memory[a]
        return -self.exploration_memory.get(angle, 0.0) * self.config.exploration_penalty_scale

    def record_heading_choice(self, angle: int):
        if angle == 0:
            return
        self.exploration_memory[angle] = self.exploration_memory.get(angle, 0.0) + 1.0

    def angle_to_cell(self, angle: int) -> int:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        normalized = (angle + max_angle) / (2.0 * max_angle)
        cell = int(normalized * self.config.spatial_memory_cells)
        return max(0, min(self.config.spatial_memory_cells - 1, cell))

    def update_spatial_memory(self, scan: Dict[int, float]):
        decay = self.config.spatial_memory_decay
        for cell in self.spatial_memory:
            self.spatial_memory[cell] *= decay
        for angle, dist in scan.items():
            cell = self.angle_to_cell(angle)
            if dist > 0.0 and dist < self.config.spatial_obstacle_threshold:
                self.spatial_memory[cell] = min(1.0, self.spatial_memory[cell] + 0.4)
            elif dist > self.config.open_space_distance:
                self.spatial_memory[cell] = max(0.0, self.spatial_memory[cell] - 0.1)

    def update_frontier_map(self, scan: Dict[int, float]):
        decay = self.config.frontier_decay
        for cell in self.frontier_map:
            self.frontier_map[cell] *= decay
        for angle, dist in scan.items():
            cell = self.angle_to_cell(angle)
            if dist > 0.0:
                if dist < self.config.caution_distance:
                    self.frontier_map[cell] = max(0.0, self.frontier_map[cell] - 0.3)
                elif dist > self.config.cruise_distance:
                    self.frontier_map[cell] = min(1.0, self.frontier_map[cell] + 0.15)

    def frontier_bonus_for_angle(self, angle: int) -> float:
        cell = self.angle_to_cell(angle)
        return self.frontier_map.get(cell, 0.5) * self.config.frontier_exploration_gain

    def is_dead_end(self, front_distance: float) -> bool:
        if front_distance > self.config.dead_end_front_threshold:
            return False
        left = self.sanitize_distance(self.last_scan.get(-45, 0.0))
        right = self.sanitize_distance(self.last_scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return False
        return left < self.config.dead_end_side_threshold and right < self.config.dead_end_side_threshold

    def get_escape_escalation_level(self, now: float) -> int:
        if now - self.last_escape_time > self.config.escape_escalation_window:
            return 0
        return min(self.escape_escalation_level, self.config.escape_escalation_levels - 1)

    def escalate_escape(self, now: float):
        self.last_escape_time = now
        self.escape_escalation_level = min(
            self.escape_escalation_level + 1,
            self.config.escape_escalation_levels
        )

    def apply_dead_end_recovery(self, heading: int, now: float) -> int:
        if now - self.dead_end_recovery_time < 1.5:
            return heading
        return heading

    def predict_collision(self, heading: int, distance: float, speed: float) -> bool:
        if speed <= 0.0 or distance <= 0.0:
            return False
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        angle_factor = abs(heading) / max_angle
        forward_component = distance * (1.0 - angle_factor * 0.5)
        predicted_distance = forward_component - speed * self.config.predictive_horizon_s * 100.0
        return predicted_distance < self.config.danger_distance + self.config.predictive_collision_margin

    def record_distance(self, speed: float, dt: float):
        self.total_distance_traveled += abs(speed) * dt * 50.0

    def estimate_complexity(self, scan: Dict[int, float]) -> float:
        if not scan:
            return 0.0
        distances = [self.sanitize_distance(d) for d in scan.values() if d > 0.0]
        if len(distances) < 2:
            return 0.0
        mean_dist = sum(distances) / len(distances)
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        return min(1.0, (variance ** 0.5) / 50.0)

    def update_complexity(self, scan: Dict[int, float]):
        complexity = self.estimate_complexity(scan)
        self.complexity_history.append(complexity)

    def get_adaptive_scan_interval(self) -> float:
        if not self.complexity_history:
            return self.config.adaptive_scan_base
        avg_complexity = sum(self.complexity_history) / len(self.complexity_history)
        multiplier = 1.0 - avg_complexity * (1.0 - self.config.complexity_scan_multiplier)
        return self.config.adaptive_scan_base * multiplier

    def check_boredom(self, heading: int, now: float) -> bool:
        if self.boredom_timer <= 0.0:
            self.boredom_timer = now
            self.last_heading_for_boredom = heading
            self.last_heading_change_time = now
            return False
        if heading != self.last_heading_for_boredom:
            if abs(heading - self.last_heading_for_boredom) > 20:
                self.consecutive_small_turns = 0
                self.boredom_timer = now
            else:
                self.consecutive_small_turns += 1
            self.last_heading_for_boredom = heading
            self.last_heading_change_time = now
            return False
        if now - self.boredom_timer > self.config.boredom_threshold_s:
            return True
        return False

    def apply_boredom_exploration(self, scan: Dict[int, float], current_heading: int) -> int:
        if not scan:
            return current_heading

        ordered_scan = tuple(
            (angle, self.sanitize_distance(distance))
            for angle, distance in sorted(scan.items())
        )
        candidate_scan = self.build_gap_headings(ordered_scan)
        positive_distances = [distance for _, distance in candidate_scan if distance > 0.0]
        if not positive_distances:
            return current_heading

        best_angle = self._choose_best_heading(
            candidate_scan,
            front_distance=max(positive_distances),
            now=None,
            escape=False,
        )
        return best_angle if best_angle != 0 else current_heading

    def smooth_heading(self, raw_heading: int) -> float:
        if not self.heading_initialized:
            self.smoothed_heading = float(raw_heading)
            self.heading_initialized = True
            return self.smoothed_heading
        alpha = self.config.heading_smoothing_alpha
        self.smoothed_heading = alpha * raw_heading + (1.0 - alpha) * self.smoothed_heading
        return self.smoothed_heading

    def record_heading_change(self, heading: int):
        self.heading_history.append(heading)

    def is_oscillating(self) -> bool:
        if len(self.heading_history) < self.config.oscillation_window:
            return False
        history = list(self.heading_history)
        sign_changes = 0
        for i in range(1, len(history)):
            if (history[i] > 0 and history[i - 1] < 0) or (history[i] < 0 and history[i - 1] > 0):
                sign_changes += 1
        return sign_changes >= self.config.oscillation_threshold

    def break_oscillation(self, current_heading: int, scan: Dict[int, float]) -> int:
        # Force a decisive turn in the direction with the most open space
        best_angle = current_heading
        best_distance = -1.0
        for angle, dist in scan.items():
            if dist > best_distance and abs(angle) > 30:
                best_distance = dist
                best_angle = angle
        if best_angle == current_heading:
            best_angle = -best_angle if best_angle != 0 else -80
        return best_angle

    def compute_scan_confidence(self, scan: Dict[int, float]) -> float:
        distances = [self.sanitize_distance(d) for d in scan.values() if d > 0.0]
        if len(distances) < 2:
            return 1.0
        mean_dist = sum(distances) / len(distances)
        if mean_dist <= 0.0:
            return 0.0
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        return min(1.0, (variance ** 0.5) / 50.0)

    def update_scan_confidence(self, scan: Dict[int, float]):
        confidence = self.compute_scan_confidence(scan)
        self.scan_confidence_history.append(confidence)

    def is_low_confidence(self) -> bool:
        if not self.scan_confidence_history:
            return False
        avg = sum(self.scan_confidence_history) / len(self.scan_confidence_history)
        return avg > self.config.scan_confidence_threshold

    def get_gap_center_steer(self, left_dist: float, right_dist: float) -> float:
        if left_dist <= 0.0 or right_dist <= 0.0:
            return 0.0
        total = left_dist + right_dist
        if total <= 0.0:
            return 0.0
        return (left_dist - right_dist) / total * self.config.gap_centering_gain

    def apply_gap_entry_slowdown(self, speed: float, front_distance: float, left_side: float, right_side: float) -> float:
        if front_distance > self.config.gap_entry_distance:
            return speed
        if left_side <= 0.0 or right_side <= 0.0:
            return speed
        min_side = min(left_side, right_side)
        if min_side < self.config.gap_entry_distance:
            tightness = 1.0 - (min_side / self.config.gap_entry_distance)
            return speed * (1.0 - tightness * self.config.gap_entry_slowdown)
        return speed

    def score_heading(
        self,
        ordered_scan: Tuple[Tuple[int, float], ...],
        index: int,
        front_distance: float,
        now: float | None = None,
        escape: bool = False,
    ) -> float:
        angle, distance = ordered_scan[index]
        if distance <= 0.0:
            return -math.inf

        max_angle = max(abs(scan_angle) for scan_angle, _ in ordered_scan) or 1
        support, neighbour_count = self.corridor_support(ordered_scan, index)
        center_weight = 1.0 - (abs(angle) / max_angle)
        turn_habit = sum(self.recent_turns) / len(self.recent_turns) if self.recent_turns else 0.0

        score = distance * (1.0 - self.config.corridor_support_weight)
        score += support * self.config.corridor_support_weight
        score += support * 0.35
        score -= max(distance - support, 0.0) * self.config.isolation_penalty
        score += max(distance - front_distance, 0.0) * self.config.distance_advantage_gain
        score -= (2 - neighbour_count) * self.config.edge_penalty * (abs(angle) / max_angle)

        clear_bonus = max(front_distance - self.config.caution_distance, 0.0)
        score += clear_bonus * self.config.forward_bias_gain * center_weight

        if front_distance <= self.config.caution_distance:
            score -= abs(angle) * 0.04
        else:
            score -= abs(angle) * 0.12

        if angle == 0 and front_distance <= self.config.caution_distance:
            score -= self.config.caution_distance + max(self.config.caution_distance - distance, 0.0)

        if abs(angle) >= 30 and support < self.config.caution_distance:
            score -= self.config.narrow_gap_penalty * (1.0 - support / self.config.caution_distance)

        if angle != 0:
            score -= math.copysign(turn_habit * self.config.exploration_bonus, angle)

        score += self.heading_memory_bonus(angle, now)

        if escape:
            score += abs(angle) * self.config.escape_angle_bias

        if self.angle_ema:
            mean_ema = sum(self.angle_ema.values()) / len(self.angle_ema)
            ema_val = self.angle_ema.get(angle, mean_ema)
            if mean_ema > 0.0:
                score += (ema_val / mean_ema - 1.0) * self.config.history_bias_gain

        if self.current_heading != 0 and not escape:
            angle_gap = abs(angle - self.current_heading)
            if angle_gap < 40:
                score += self.config.heading_momentum_gain * (1.0 - angle_gap / 40.0)

        score += self.gap_bonus_for_angle(angle, ordered_scan)

        score += self.exploration_penalty(angle)

        rate = self.get_angle_approach_rate(angle)
        if rate > 0.0 and distance < self.config.caution_distance:
            score -= rate * 0.5

        cell = self.angle_to_cell(angle)
        spatial_obstacle = self.spatial_memory.get(cell, 0.0)
        if spatial_obstacle > 0.1:
            score -= spatial_obstacle * 15.0

        score += self.frontier_bonus_for_angle(angle)

        if self.path_commitment_heading != 0:
            angle_gap = abs(angle - self.path_commitment_heading)
            if angle_gap < 25:
                score += self.config.path_commitment_bonus

        return score

    def _choose_best_heading(
        self,
        ordered_scan: Tuple[Tuple[int, float], ...],
        front_distance: float,
        now: float | None = None,
        escape: bool = False,
    ) -> int:
        best_angle = 0
        best_score = -math.inf

        for index, (angle, distance) in enumerate(ordered_scan):
            if distance <= 0.0:
                continue

            score = self.score_heading(
                ordered_scan=ordered_scan,
                index=index,
                front_distance=front_distance,
                now=now,
                escape=escape,
            )

            if score > best_score or (math.isclose(score, best_score) and abs(angle) < abs(best_angle)):
                best_score = score
                best_angle = angle

        return best_angle

    def select_heading(self, scan: Dict[int, float], front_distance: float, now: float | None = None, escape: bool = False) -> int:
        if not scan:
            return 0

        front_distance = self.sanitize_distance(front_distance)
        ordered_scan = tuple(
            (angle, self.sanitize_distance(distance))
            for angle, distance in sorted(scan.items())
        )
        candidate_scan = self.build_gap_headings(ordered_scan) if self.should_use_gap_headings(front_distance) else ordered_scan
        return self._choose_best_heading(candidate_scan, front_distance, now=now, escape=escape)

    def remember_heading(self, heading: int, front_distance: float, now: float):
        if front_distance <= self.config.caution_distance or abs(heading) < 30:
            return
        self.target_heading = heading
        self.target_heading_time = now

    def plan(self, front_distance: float, now: float) -> MotionCommand:
        front_distance = self.observe_front(front_distance)
        approach_rate = self.track_approach_rate(front_distance, now)
        heading = self.select_heading(self.last_scan, front_distance, now=now)

        if front_distance <= self.config.danger_distance:
            self.current_speed = 0.0
            self.escalate_escape(now)
            escalation = self.get_escape_escalation_level(now)
            heading = self.select_heading(self.last_scan, front_distance, now=now, escape=True)
            if heading == 0:
                heading = -80 if sum(self.recent_turns) >= 0 else 80
            self.target_heading = 0
            self.target_heading_time = -math.inf
            self.stuck_escape_times.append(now)
            self.path_commitment_heading = 0
            self.path_commitment_time = -math.inf
            return MotionCommand(
                mode="escape",
                left_speed=-self.config.escape_reverse_speed,
                right_speed=-self.config.escape_reverse_speed,
                heading=heading,
                colour=(255, 48, 0),
            )

        if self.is_dead_end(front_distance):
            self.dead_end_recovery_time = now
            self.escalate_escape(now)
            heading = -80 if sum(self.recent_turns) >= 0 else 80
            self.current_speed = 0.0
            self.path_commitment_heading = 0
            self.path_commitment_time = -math.inf
            return MotionCommand(
                mode="dead_end_recovery",
                left_speed=-self.config.escape_reverse_speed * 0.6,
                right_speed=-self.config.escape_reverse_speed * 0.6,
                heading=heading,
                colour=(255, 128, 0),
            )

        if now - self.dead_end_recovery_time < 1.5:
            recovery_progress = (now - self.dead_end_recovery_time) / 1.5
            turn_speed = self.config.escape_turn_speed * min(1.0, recovery_progress * 2.0)
            heading = -80 if sum(self.recent_turns) >= 0 else 80
            if heading < 0:
                left_s = -0.2 * (1.0 - recovery_progress)
                right_s = turn_speed
            else:
                left_s = turn_speed
                right_s = -0.2 * (1.0 - recovery_progress)
            self.current_speed = 0.0
            return MotionCommand(
                mode="dead_end_recovery",
                left_speed=left_s,
                right_speed=right_s,
                heading=heading,
                colour=(255, 128, 0),
            )

        if now - self.path_commitment_time > self.config.path_commitment_s:
            self.path_commitment_heading = heading
            self.path_commitment_time = now

        if self.predict_collision(heading, front_distance, self.current_speed):
            heading = self.select_heading(self.last_scan, front_distance, now=now, escape=True)
            if heading == 0:
                heading = -80 if sum(self.recent_turns) >= 0 else 80

        if self.check_boredom(heading, now):
            heading = self.apply_boredom_exploration(self.last_scan, heading)
            self.boredom_timer = now

        if self.is_oscillating():
            heading = self.break_oscillation(heading, self.last_scan)

        self.record_heading_change(heading)

        positive_values = [v for v in self.last_scan.values() if v > 0.0]
        open_space = bool(positive_values) and all(
            v > self.config.open_space_distance for v in positive_values
        )
        if front_distance > self.config.cruise_distance and open_space:
            base_speed = self.config.open_space_speed
        elif front_distance <= self.config.cruise_distance:
            base_speed = self.config.cautious_speed
        else:
            base_speed = self.config.cruise_speed
        delta = base_speed - self.current_speed
        if delta > 0:
            speed = self.current_speed + min(delta, self.config.speed_accel_rate)
        else:
            speed = self.current_speed + max(delta, -self.config.speed_decel_rate)

        if approach_rate > 0.0 and front_distance < self.config.caution_distance:
            speed = max(0.0, speed - approach_rate * self.config.approach_rate_decel_gain)

        if front_distance < self.config.proximity_urgency_distance:
            urgency_factor = 1.0 - (front_distance / self.config.proximity_urgency_distance)
            speed = max(0.0, speed * (1.0 - urgency_factor * self.config.proximity_urgency_gain))

        left_side = self.last_scan.get(-45, 0.0)
        right_side = self.last_scan.get(45, 0.0)
        speed = self.apply_gap_entry_slowdown(speed, front_distance, left_side, right_side)

        self.current_speed = speed
        steer = 0.0
        max_angle = max(abs(angle) for angle in self.config.scan_angles) or 1

        if heading != 0:
            steer = max(-1.0, min(1.0, heading / max_angle))
            if front_distance <= self.config.caution_distance:
                steer *= 1.20

        # Side-proximity correction: nudge away from closer wall when heading is roughly straight
        if abs(heading) < 30 and front_distance > self.config.caution_distance:
            left_side = self.last_scan.get(-45, 0.0)
            right_side = self.last_scan.get(45, 0.0)
            if left_side > 0.0 and right_side > 0.0:
                total = left_side + right_side
                if total > 0.0:
                    # positive when right > left (left wall closer) → steer right (positive = right)
                    side_corr = (right_side - left_side) / total * self.config.side_correction_gain
                    steer = max(-1.0, min(1.0, steer + side_corr))

        # Wall-following mode: when in a corridor (both sides detected, front clear), maintain parallel alignment
        left_side = self.last_scan.get(-45, 0.0)
        right_side = self.last_scan.get(45, 0.0)
        in_corridor = (
            left_side > 0.0 and right_side > 0.0
            and left_side < self.config.wall_following_threshold
            and right_side < self.config.wall_following_threshold
            and front_distance > self.config.caution_distance
        )
        if in_corridor:
            imbalance = (left_side - right_side) / (left_side + right_side)
            wall_follow_steer = -imbalance * self.config.wall_following_gain
            steer = max(-1.0, min(1.0, steer + wall_follow_steer))

        # Gap centering: when entering a gap between obstacles, actively center the car
        if front_distance < self.config.gap_entry_distance and abs(heading) < 45:
            gap_steer = self.get_gap_center_steer(left_side, right_side)
            steer = max(-1.0, min(1.0, steer + gap_steer))

        # Apply temporal heading smoothing to reduce jitter
        effective_heading = self.smooth_heading(heading)

        left_speed = max(-1.0, min(1.0, speed * (1.0 + steer * self.config.steer_gain)))
        right_speed = max(-1.0, min(1.0, speed * (1.0 - steer * self.config.steer_gain)))

        colour = (32, 160, 255) if abs(effective_heading) >= 45 else (0, 255, 64)
        if front_distance <= self.config.caution_distance:
            colour = (255, 180, 0)

        self.remember_heading(heading, front_distance, now)
        self.current_heading = heading
        self.record_heading_choice(heading)
        self.record_distance(speed, self.config.loop_delay_s)

        return MotionCommand(
            mode="drive",
            left_speed=left_speed,
            right_speed=right_speed,
            heading=int(round(effective_heading)),
            colour=colour,
        )


def perform_scan(tbot, controller: AutonomousCarController) -> Dict[int, float]:
    scan = {}
    for angle in controller.config.scan_angles:
        tbot.set_servo_angle(angle)
        time.sleep(controller.config.scan_pause_s)
        scan[angle] = tbot.read_distance(
            timeout=controller.config.scan_timeout_ms,
            samples=controller.config.scan_samples,
        )
    tbot.set_servo_angle(0)
    time.sleep(controller.config.scan_pause_s)
    controller.update_scan(scan, time.monotonic())
    return scan


def apply_command(tbot, controller: AutonomousCarController, command: MotionCommand):
    tbot.fill_underlighting(command.colour)

    if command.mode == "escape":
        controller.note_turn(command.heading)
        escalation = controller.get_escape_escalation_level(time.monotonic())
        reverse_duration = controller.config.escape_reverse_s * (1.0 + escalation * 0.3)
        turn_duration = controller.config.escape_turn_s * (1.0 + escalation * 0.4)
        tbot.backward(controller.config.escape_reverse_speed)
        time.sleep(reverse_duration)
        if command.heading < 0:
            tbot.turn_left(controller.config.escape_turn_speed)
        else:
            tbot.turn_right(controller.config.escape_turn_speed)
        time.sleep(turn_duration)
        tbot.stop()
        if controller.is_stuck(time.monotonic()):
            controller.stuck_escape_times.clear()
            spin_duration = controller.config.stuck_spin_s * (1.0 + escalation * 0.5)
            if sum(controller.recent_turns) >= 0:
                tbot.turn_left(controller.config.escape_turn_speed)
            else:
                tbot.turn_right(controller.config.escape_turn_speed)
            time.sleep(spin_duration)
            tbot.stop()
        controller.last_scan_time = -math.inf
        return

    if command.mode == "dead_end_recovery":
        controller.note_turn(command.heading)
        tbot.set_motor_speeds(command.left_speed, command.right_speed)
        time.sleep(controller.config.dead_end_recovery_turn_s)
        tbot.stop()
        controller.last_scan_time = -math.inf
        return

    if abs(command.heading) >= 30:
        controller.note_turn(command.heading)

    tbot.set_motor_speeds(command.left_speed, command.right_speed)


def main():
    from trilobot import BUTTON_A, Trilobot

    print("Trilobot Example: Autonomous Car\n")

    tbot = Trilobot()
    controller = AutonomousCarController(AutonomousCarConfig())

    try:
        tbot.initialise_servo()
        tbot.set_servo_angle(0)
        time.sleep(0.25)
        perform_scan(tbot, controller)

        while not tbot.read_button(BUTTON_A):
            now = time.monotonic()
            front_distance = tbot.read_distance(
                timeout=controller.config.front_timeout_ms,
                samples=controller.config.front_samples,
            )

            if controller.last_scan:
                controller.last_scan[0] = controller.sanitize_distance(front_distance)

            if controller.should_scan(front_distance, now):
                perform_scan(tbot, controller)
                front_distance = controller.last_scan.get(0, front_distance)

            command = controller.plan(front_distance, now)
            apply_command(tbot, controller, command)
            time.sleep(controller.config.loop_delay_s)
    finally:
        tbot.stop()
        tbot.clear_underlighting()
        try:
            tbot.set_servo_angle(0)
            time.sleep(0.1)
            tbot.disable_servo()
        except Exception:
            pass
        tbot.cleanup()


if __name__ == "__main__":
    main()
