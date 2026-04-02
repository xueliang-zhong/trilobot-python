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
    last_scan_time: float = field(default=-math.inf)
    target_heading: int = field(default=0)
    target_heading_time: float = field(default=-math.inf)
    current_speed: float = field(default=0.0)

    def __post_init__(self):
        self.front_history = deque(maxlen=self.config.front_history_size)
        self.recent_turns = deque(maxlen=self.config.recent_turn_memory)
        self.stuck_escape_times = deque(maxlen=self.config.stuck_escape_count + 1)

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
        positive_scan = [v for v in self.last_scan.values() if v > 0.0]
        was_open = bool(positive_scan) and all(v > self.config.open_space_distance for v in positive_scan)
        proactive_interval = self.config.open_space_scan_s if was_open else self.config.proactive_scan_s
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

        return score

    def select_heading(self, scan: Dict[int, float], front_distance: float, now: float | None = None, escape: bool = False) -> int:
        if not scan:
            return 0

        front_distance = self.sanitize_distance(front_distance)
        ordered_scan = tuple(
            (angle, self.sanitize_distance(distance))
            for angle, distance in sorted(scan.items())
        )

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

    def remember_heading(self, heading: int, front_distance: float, now: float):
        if front_distance <= self.config.caution_distance or abs(heading) < 30:
            return
        self.target_heading = heading
        self.target_heading_time = now

    def plan(self, front_distance: float, now: float) -> MotionCommand:
        front_distance = self.observe_front(front_distance)
        heading = self.select_heading(self.last_scan, front_distance, now=now)

        if front_distance <= self.config.danger_distance:
            self.current_speed = 0.0
            heading = self.select_heading(self.last_scan, front_distance, now=now, escape=True)
            if heading == 0:
                heading = -80 if sum(self.recent_turns) >= 0 else 80
            self.target_heading = 0
            self.target_heading_time = -math.inf
            self.stuck_escape_times.append(now)
            return MotionCommand(
                mode="escape",
                left_speed=-self.config.escape_reverse_speed,
                right_speed=-self.config.escape_reverse_speed,
                heading=heading,
                colour=(255, 48, 0),
            )

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
        self.current_speed = speed
        steer = 0.0
        max_angle = max(abs(angle) for angle in self.config.scan_angles) or 1

        if heading != 0:
            steer = max(-1.0, min(1.0, heading / max_angle))
            if front_distance <= self.config.caution_distance:
                steer *= 1.20

        left_speed = max(-1.0, min(1.0, speed * (1.0 + steer * self.config.steer_gain)))
        right_speed = max(-1.0, min(1.0, speed * (1.0 - steer * self.config.steer_gain)))

        colour = (32, 160, 255) if abs(heading) >= 45 else (0, 255, 64)
        if front_distance <= self.config.caution_distance:
            colour = (255, 180, 0)

        self.remember_heading(heading, front_distance, now)

        return MotionCommand(
            mode="drive",
            left_speed=left_speed,
            right_speed=right_speed,
            heading=heading,
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
        tbot.backward(controller.config.escape_reverse_speed)
        time.sleep(controller.config.escape_reverse_s)
        if command.heading < 0:
            tbot.turn_left(controller.config.escape_turn_speed)
        else:
            tbot.turn_right(controller.config.escape_turn_speed)
        time.sleep(controller.config.escape_turn_s)
        tbot.stop()
        if controller.is_stuck(time.monotonic()):
            controller.stuck_escape_times.clear()
            if sum(controller.recent_turns) >= 0:
                tbot.turn_left(controller.config.escape_turn_speed)
            else:
                tbot.turn_right(controller.config.escape_turn_speed)
            time.sleep(controller.config.stuck_spin_s)
            tbot.stop()
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
