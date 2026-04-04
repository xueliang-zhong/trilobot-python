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
import os
import time
from colorsys import hsv_to_rgb
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
    exploration_spread_degrees: float = 35.0  # angular radius over which recent headings suppress nearby choices
    per_angle_rate_window: int = 3        # samples for per-angle approach rate tracking
    spatial_memory_cells: int = 5          # number of angular sectors in spatial memory
    spatial_memory_decay: float = 0.97     # decay rate for spatial memory per loop cycle
    spatial_obstacle_threshold: float = 25.0  # distance below which a cell is marked as blocked
    frontier_exploration_gain: float = 6.0  # score bonus for under-explored directions
    frontier_decay: float = 0.95           # decay rate for frontier memory per loop cycle
    dead_end_front_threshold: float = 35.0  # max front distance to trigger dead end check
    dead_end_side_threshold: float = 40.0   # max side distance to confirm dead end
    dead_end_recovery_turn_s: float = 0.45  # duration of dead end recovery turn
    push_min_side: float = 50.0             # both sides must be > this (cm) to attempt a brave push
    push_speed: float = 0.90               # forward speed during brave push
    push_duration_s: float = 0.9           # how long to push before rescanning
    push_cooldown_s: float = 6.0           # minimum gap between push attempts
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
    danger_heatmap_decay: float = 0.995       # decay per loop cycle (~14s half-life at 60ms/loop)
    danger_penalty_scale: float = 22.0        # score penalty per unit of danger memory
    danger_score_escape: float = 1.0          # danger recorded per escape/dead-end event
    danger_score_push: float = 0.5            # danger recorded per brave-push attempt
    follow_target_distance: float = 40.0      # cm to maintain behind followed object
    follow_deadband: float = 6.0              # ±cm tolerance before correcting
    follow_max_speed: float = 0.55            # top speed when chasing
    follow_reverse_speed: float = 0.28        # speed when backing off
    follow_kp: float = 0.018                  # proportional gain (speed per cm of error)
    stress_escape_gain: float = 0.25          # stress added per escape / dead-end
    stress_push_gain: float = 0.10            # stress added per brave push
    stress_decay: float = 0.998              # per-loop decay (~34s half-life)
    pounce_trigger_distance: float = 32.0     # max front distance to arm a predatory lane-change burst
    pounce_min_clearance_gain: float = 24.0   # minimum extra side clearance required to trigger a pounce
    pounce_min_heading: int = 30              # only commit to decisive side headings
    pounce_speed: float = 0.92                # burst speed during peek-and-pounce
    pounce_commit_s: float = 0.55             # how long to hold the chosen burst heading
    pounce_steer_gain: float = 1.35           # stronger steering than normal drive mode
    flow_memory_window: int = 8               # recent headings to track flow quality
    flow_memory_gain: float = 6.0             # score bonus per unit of flow memory
    flow_memory_decay: float = 0.93           # decay per loop cycle
    flow_quality_threshold: float = 0.3       # minimum front improvement to count as good flow
    danger_zone_cells: int = 9                # angular sectors for danger zone memory
    danger_zone_penalty: float = 20.0         # score penalty per unit of danger zone memory
    danger_zone_decay: float = 0.98           # decay per loop cycle
    danger_zone_event_score: float = 0.6      # danger added per escape/dead-end event
    option_value_weight: float = 0.4          # weight of future-option scoring in lookahead
    option_value_angle_range: int = 60        # angular range to count as "options available"
    side_closing_window: int = 4              # recent scans to detect side-closing pattern
    side_closing_threshold: float = 12.0      # cm decrease to trigger side-closing alert
    side_closing_slowdown: float = 0.25       # speed multiplier when sides are closing
    side_closing_scan_urgency: float = 0.5    # scan interval multiplier when sides closing
    novelty_window: int = 12                  # recent headings for novelty detection
    novelty_angular_radius: float = 25.0      # degrees within which a heading counts as "seen"
    novelty_bonus: float = 4.0                # score bonus for choosing a novel heading
    novelty_min_flow_entries: int = 3         # require this many flow entries before novelty applies
    quip_cooldown_s: float = 2.5              # minimum seconds between humour quips
    quip_history_size: int = 6                # recent quips to avoid repeating

    lookahead_steps: int = 3                  # how many future steps to simulate
    lookahead_distance_per_step: float = 18.0 # cm assumed per lookahead step at base speed
    lookahead_angle_penalty: float = 0.6      # penalty per degree of heading change in lookahead
    lookahead_obstacle_margin: float = 12.0   # extra cm margin when simulating future collisions

    terrain_classify_window: int = 8          # scan samples for terrain classification
    terrain_open_space_threshold: float = 80.0  # avg distance to classify as open terrain
    terrain_cluttered_threshold: float = 35.0  # avg distance to classify as cluttered terrain
    terrain_corridor_width_max: float = 55.0   # max side-to-side distance to classify as corridor
    terrain_open_speed_multiplier: float = 1.15  # speed boost in open terrain
    terrain_cluttered_speed_multiplier: float = 0.82  # speed reduction in cluttered terrain
    terrain_corridor_speed_multiplier: float = 1.05  # slight boost in corridors (predictable)

    curvature_window: int = 5                 # recent headings for curvature calculation
    curvature_speed_gain: float = 0.25        # speed reduction per unit of curvature
    curvature_sharp_threshold: float = 40.0   # degrees of heading change to trigger sharp turn slowdown
    curvature_brake_distance: float = 25.0    # cm front distance to trigger curvature braking

    corner_detection_window: int = 4          # scans to confirm a corner pattern
    corner_front_threshold: float = 30.0      # max front distance to suspect a corner
    corner_side_asymmetry: float = 20.0       # min difference between left/right to confirm corner
    corner_turn_ahead_gain: float = 0.35      # bias toward turning into the open side of a corner
    corner_slowdown_factor: float = 0.70      # speed multiplier when negotiating a corner

    min_gap_width: float = 22.0               # minimum estimated gap width (cm) to attempt passage
    gap_width_safety_margin: float = 8.0      # extra cm subtracted from gap width estimate
    gap_width_side_blend: float = 0.60        # weight for side readings in gap width estimate
    gap_commitment_bonus: float = 15.0        # score bonus when gap is confirmed wide enough
    gap_reject_penalty: float = 25.0          # score penalty when gap is too narrow

    obstacle_density_window: int = 6          # recent scans for obstacle density calculation
    obstacle_density_threshold: float = 0.50  # ratio of blocked cells to trigger high density
    density_speed_reduction: float = 0.20     # speed reduction in high-density areas
    density_scan_interval_multiplier: float = 0.7  # scan more often in dense areas

    path_confidence_window: int = 4           # recent plans for path confidence tracking
    path_confidence_decay: float = 0.90       # decay per loop cycle for path confidence
    low_confidence_caution_distance: float = 35.0  # extra caution distance when path confidence is low

    predictive_path_length: int = 5           # number of future positions to predict
    predictive_turn_cost: float = 0.15        # speed cost per degree of predicted turn
    predictive_clearance_bonus: float = 0.10  # speed bonus per cm of predicted clearance

    adaptive_scan_urgency_multiplier: float = 0.4  # scan interval multiplier when urgency is high
    adaptive_scan_urgency_distance: float = 25.0   # distance threshold for urgent scanning

    momentum_turn_threshold: float = 35.0     # degrees of sustained heading to build momentum
    momentum_speed_bonus: float = 0.08        # speed bonus when moving with momentum
    momentum_decay: float = 0.95              # momentum decay per loop cycle

    side_obstacle_proximity_distance: float = 30.0  # cm to consider side obstacle as relevant
    side_obstacle_avoidance_gain: float = 0.20  # steer gain to avoid side obstacles
    preemptive_side_avoid_distance: float = 45.0  # cm to start preemptive side avoidance

    multi_angle_escape_angles: Tuple[int, ...] = (-60, -30, 0, 30, 60)  # angles to try during escape
    escape_angle_preference_bonus: float = 8.0  # bonus for escape angles with more clearance

    drift_correction_gain: float = 0.08       # proportional gain for drift correction
    drift_correction_window: int = 3          # recent headings for drift estimation
    drift_threshold: float = 15.0             # degrees of sustained offset to trigger correction
    personality_adapt_rate: float = 0.03      # how fast personality shifts per loop cycle
    personality_caution_weight: float = 0.4   # weight of recent caution in personality
    personality_stress_weight: float = 0.3    # weight of stress in personality
    personality_success_weight: float = 0.3   # weight of successful navigation in personality
    heading_hysteresis_threshold: float = 5.0  # minimum score difference to change heading
    heading_hysteresis_window: int = 3          # cycles to maintain heading before allowing switch
    obstacle_velocity_window: int = 6           # samples for tracking obstacle movement
    obstacle_velocity_threshold: float = 8.0    # cm/s to classify as moving obstacle
    moving_obstacle_caution_multiplier: float = 1.4  # extra caution when obstacles are moving
    escape_arc_radius: float = 0.6              # arc curvature during escape (0=straight, 1=tight)
    escape_arc_duration: float = 0.35           # duration of arc phase in escape
    environmental_memory_size: int = 20         # recent notable positions to remember
    environmental_story_window: float = 30.0    # seconds of recent history for storytelling
    narrative_cooldown_s: float = 12.0          # minimum seconds between narrative quips
    irony_chance: float = 0.08                  # probability of ironic/sarcastic quip
    running_commentary_interval: float = 8.0    # seconds between running commentary
    side_corridor_open_threshold: float = 55.0  # side distance to detect a side corridor
    side_corridor_asymmetry_min: float = 20.0   # min left-right difference for side corridor
    side_corridor_bonus: float = 14.0           # score bonus for heading toward side corridor
    side_corridor_memory_s: float = 4.0         # how long to remember a side corridor
    escape_outcome_prediction_steps: int = 3    # steps to simulate before committing to escape
    escape_prediction_clearance_min: float = 18.0  # min clearance per step for good escape
    persistence_window: int = 8               # number of recent scans to track obstacle persistence
    persistence_obstacle_threshold: float = 0.6  # persistence score above which obstacle is "static"
    persistence_static_penalty: float = 15.0   # extra score penalty for persistent obstacles
    corridor_exit_front_threshold: float = 70.0  # front distance to detect corridor exit
    corridor_exit_min_side: float = 50.0       # min side distance to confirm corridor exit
    corridor_exit_bonus: float = 20.0          # score bonus for heading toward corridor exit
    corridor_exit_anticipation_s: float = 3.0  # how long to remember a detected exit
    recovery_stage_duration: float = 2.0       # seconds per recovery stage before escalating
    recovery_wiggle_speed: float = 0.35        # speed during wiggling recovery
    recovery_wiggle_s: float = 0.25            # duration of each wiggle turn
    escape_direction_learning_rate: float = 0.3  # how much to weight new escape outcomes
    dead_end_prediction_window: int = 3           # scans to confirm a dead-end pattern forming
    dead_end_prediction_front_max: float = 30.0    # max front distance to suspect dead-end forming
    dead_end_prediction_side_max: float = 35.0     # max side distance to confirm dead-end forming
    dead_end_prediction_avoidance: float = 0.35    # heading nudge away from predicted dead-end
    dead_end_prediction_slowdown: float = 0.25     # speed reduction when dead-end predicted
    obstacle_shape_window: int = 5                  # scans for obstacle shape inference
    shape_wall_front_threshold: float = 25.0        # max front to suspect wall
    shape_wall_side_tolerance: float = 15.0         # max left-right difference for wall
    shape_corner_side_gap: float = 20.0             # min side difference for corner
    shape_isolated_max_width: float = 30.0          # max front span for isolated object
    success_map_cells: int = 8                      # angular sectors for success tracking
    success_map_decay: float = 0.96                 # decay per loop cycle
    success_map_gain: float = 10.0                  # score bonus per unit of success memory
    success_map_quality_threshold: float = 0.1      # min front improvement to count as success
    terrain_memory_cells: int = 5                   # angular sectors for terrain memory
    terrain_memory_decay: float = 0.98              # decay per loop cycle
    terrain_memory_open_bonus: float = 5.0          # score bonus for heading toward remembered open terrain
    achievement_window_size: int = 12               # recent quips to check for achievement triggers
    self_aware_quip_cooldown: float = 8.0           # seconds between self-aware/meta quips

    tactical_patience_duration: float = 0.6         # seconds to pause and reassess in ambiguous situations
    tactical_patience_front_max: float = 35.0       # max front distance to trigger patience
    tactical_patience_side_min: float = 40.0        # min side clearance to allow patience (not trapped)
    patience_reassess_bonus: float = 15.0           # score bonus for heading chosen after patience pause

    junction_front_open_threshold: float = 50.0     # front distance to suspect a junction
    junction_side_open_threshold: float = 60.0      # side distance to confirm junction
    junction_side_asymmetry_min: float = 25.0       # min left-right difference for T-junction
    junction_unexplored_bonus: float = 18.0         # bonus for unexplored junction direction
    junction_corner_preference: float = 0.6         # weight toward more open side at junction

    near_miss_distance: float = 18.0                # distance threshold for near-miss detection
    near_miss_window: int = 6                       # recent samples to track near-misses
    near_miss_caution_multiplier: float = 1.25      # extra caution distance multiplier after near-miss cluster
    near_miss_learned_avoid_gain: float = 10.0      # score penalty for angles involved in recent near-misses

    exploration_strategy: str = "balanced"          # current strategy: edge_follow/center/balanced/random
    strategy_adapt_window: int = 15                 # samples before re-evaluating strategy
    edge_follow_side_preference: int = -1           # which side to follow (-1=left, 1=right)
    edge_follow_gain: float = 0.15                  # steering gain toward preferred wall
    center_lane_gain: float = 0.10                  # steering gain to stay centered between walls
    strategy_edge_reward: float = 0.05              # reward for successful edge-following
    strategy_center_reward: float = 0.05            # reward for successful center-lane navigation
    strategy_random_turn_interval: float = 8.0      # seconds between random exploration turns

    humour_tactical: bool = True                    # enable tactical commentary
    humour_junction: bool = True                    # enable junction-specific quips
    humour_situational: bool = True                 # enable situational awareness quips
    humour_meta_frequency: float = 0.15             # probability multiplier for meta-humour

    scoring_term_learning_rate: float = 0.05        # how fast scoring weights adapt per decision
    scoring_weight_floor: float = 0.3               # minimum weight multiplier for any term
    scoring_weight_ceiling: float = 2.5             # maximum weight multiplier for any term
    scoring_reward_window: int = 20                 # recent decisions to evaluate term quality

    obstacle_cluster_window: int = 8                # recent scans for cluster detection
    obstacle_cluster_blocked_ratio: float = 0.60    # ratio of blocked angles to trigger cluster
    obstacle_cluster_tight_threshold: float = 30.0  # max avg distance in cluster to call it "tight"
    cluster_caution_multiplier: float = 1.3         # extra caution in tight clusters
    cluster_scan_multiplier: float = 0.5            # scan more often in clusters
    cluster_speed_reduction: float = 0.25           # speed reduction in tight clusters

    flow_state_window: int = 25                     # loops to evaluate flow state
    flow_state_success_threshold: float = 0.60      # ratio of good decisions for "flow"
    flow_state_struggle_threshold: float = 0.30     # ratio of bad decisions for "struggle"
    flow_state_speed_bonus: float = 0.06            # speed boost when in flow
    flow_state_caution_reduction: float = 0.85      # caution multiplier when in flow
    flow_state_scan_extension: float = 1.15         # scan less often when in flow
    struggle_caution_multiplier: float = 1.2        # extra caution when struggling
    struggle_scan_urgency: float = 0.7              # scan more often when struggling

    fortune_teller_cooldown: float = 10.0           # seconds between fortune predictions
    fortune_window: int = 12                        # recent history for prediction basis

    # Gen18: coverage map
    coverage_map_cells: int = 16                    # angular sectors for coverage tracking
    coverage_map_decay: float = 0.999               # per-loop decay for coverage memory
    coverage_novelty_bonus: float = 10.0            # score bonus for heading toward unvisited sectors
    coverage_revisit_penalty: float = -4.0          # score penalty for revisiting heavily-covered sectors
    coverage_map_visit_gain: float = 0.15           # how much to increment coverage per loop in a sector

    # Gen18: frontier-based exploration
    frontier_cell_count: int = 12                   # angular sectors for frontier detection
    frontier_exploration_bonus: float = 15.0        # score bonus for heading toward frontier boundary
    frontier_decay: float = 0.998                   # per-loop decay for frontier memory
    frontier_open_threshold: float = 60.0           # distance to mark sector as "explored open"
    frontier_blocked_threshold: float = 25.0        # distance to mark sector as "blocked"

    # Gen18: straight-line preference
    straight_line_bonus: float = 8.0                # score bonus for maintaining straight heading
    straight_line_duration_threshold: float = 2.0   # seconds of straight driving to activate bonus
    straight_line_max_bonus: float = 20.0           # maximum straight-line bonus
    directional_persistence_gain: float = 0.05      # how much persistence builds per second going straight
    directional_persistence_max: float = 1.0        # max directional persistence value
    directional_persistence_decay: float = 0.98     # decay when not going straight

    # Gen18: emotional moves
    victory_dance_duration: float = 0.6             # seconds for victory dance animation
    victory_dance_wiggles: int = 3                  # number of wiggles in victory dance
    happy_wiggle_duration: float = 0.4              # seconds for happy wiggle
    frustrated_shimmy_duration: float = 0.5         # seconds for frustrated shimmy
    confident_strut_duration: float = 0.8           # seconds for confident strut burst
    confident_strut_speed: float = 0.95             # speed during confident strut
    emotional_move_cooldown: float = 8.0            # minimum seconds between emotional moves

    # Gen18: mood-based lighting
    mood_celebration_duration: float = 2.0          # seconds to show celebration lighting
    mood_celebration_pulse_rate: float = 6.0        # pulse rate during celebration
    mood_stress_wash_intensity: float = 0.6         # how much stress washes colors toward red
    mood_personality_color_influence: float = 0.3   # how much personality affects base hue
    mood_rainbow_cycle_rate: float = 0.5            # hue rotation speed during rainbow mood
    mood_breath_rate_base: float = 2.0              # base breathing rate for LEDs
    mood_breath_rate_speed_mod: float = 4.0         # breathing rate modifier based on speed

    # Gen19: 2D grid exploration map
    grid_map_size: int = 12                          # NxN grid cells for spatial exploration
    grid_cell_size_cm: float = 30.0                  # physical size of each grid cell in cm
    grid_novelty_bonus: float = 18.0                 # score bonus for unvisited grid cells
    grid_revisit_penalty: float = -6.0               # score penalty for revisiting occupied cells
    grid_decay: float = 0.9995                       # per-loop decay for grid freshness
    grid_frontier_bonus: float = 25.0                # bonus for heading toward grid frontier

    # Gen19: systematic spiral exploration
    spiral_exploration_interval: float = 30.0        # seconds between spiral exploration bursts
    spiral_duration: float = 4.0                     # how long a spiral burst lasts
    spiral_speed: float = 0.55                       # speed during spiral exploration
    spiral_turn_rate: float = 0.35                   # turn rate during spiral

    # Gen19: enhanced straight-line preference
    straight_line_max_bonus: float = 30.0            # maximum straight-line bonus (was 20.0)
    straight_line_duration_threshold: float = 1.5    # seconds to activate bonus (was 2.0)
    path_memory_window: int = 20                     # recent headings to track path smoothness
    path_smoothness_bonus: float = 8.0               # bonus for smooth continuous paths
    straight_line_grip_gain: float = 0.08            # how aggressively to maintain straight heading

    # Gen19: discovery excitement system
    discovery_cooldown_s: float = 15.0               # minimum time between discovery events
    discovery_move_duration: float = 0.8             # duration of discovery celebration move
    discovery_light_duration: float = 3.0            # how long discovery lighting lasts

    # Gen19: enhanced emotional moves
    curious_tilt_duration: float = 0.6               # duration of curious tilt
    excited_bounce_duration: float = 0.5             # duration of excited bounce
    contemplative_circle_duration: float = 1.0       # duration of contemplative circle
    celebration_spin_duration: float = 0.8           # duration of celebration spin
    surprise_shake_duration: float = 0.4             # duration of surprise shake
    emotional_move_cooldown: float = 6.0             # minimum seconds between emotional moves (was 8.0)

    # Gen19: enhanced mood lighting
    mood_aurora_duration: float = 4.0                # duration of aurora lighting effect
    mood_heartbeat_duration: float = 2.0             # duration of heartbeat lighting effect
    mood_rainbow_wave_duration: float = 5.0          # duration of rainbow wave effect
    mood_rainbow_wave_rate: float = 3.0              # hue rotation speed for rainbow wave
    mood_per_led_emotion: bool = True                # enable per-LED emotional expressions

    # Gen20: waypoint/landmark memory
    landmark_memory_size: int = 20                    # max landmarks to remember
    landmark_detection_front_threshold: float = 90.0  # front distance to register a landmark
    landmark_novelty_bonus: float = 22.0              # bonus for heading toward unvisited landmarks
    landmark_visited_penalty: float = -8.0            # penalty for heading toward visited landmarks
    landmark_decay: float = 0.999                     # per-loop decay for landmark salience
    landmark_min_spacing_cm: float = 50.0             # minimum distance between landmarks

    # Gen20: enhanced straight-line preference
    straight_line_heading_lock_bonus: float = 35.0    # maximum bonus for locked-on straight heading
    straight_lock_duration_threshold: float = 1.0     # seconds to activate heading lock
    straight_line_turn_penalty: float = 12.0          # penalty per 10 degrees of turn away from straight
    straight_line_grip_gain: float = 0.12             # stronger grip correction for straight paths
    straight_line_recovery_bonus: float = 15.0        # bonus for returning to a straight path after turning

    # Gen20: directed frontier exploration
    frontier_target_heading: int = 0                  # current frontier target heading
    frontier_target_expiry: float = 0.0               # when current frontier target expires
    frontier_target_cooldown: float = 10.0            # seconds before selecting a new frontier target
    frontier_directed_bonus: float = 28.0             # bonus for heading toward selected frontier target

    # Gen20: new emotional moves
    nervous_creep_duration: float = 0.8               # duration of nervous creep
    triumphant_arc_duration: float = 1.0              # duration of triumphant arc
    confused_figure8_duration: float = 1.2            # duration of confused figure-8
    relaxed_cruise_duration: float = 1.5              # duration of relaxed cruise
    alert_scan_duration: float = 0.6                  # duration of alert scan
    happy_wiggle_duration: float = 0.4                # seconds for happy wiggle
    frustrated_shimmy_duration: float = 0.5           # seconds for frustrated shimmy
    confident_strut_duration: float = 0.8             # seconds for confident strut burst
    confident_strut_speed: float = 0.95               # speed during confident strut
    emotional_move_cooldown: float = 5.0              # minimum seconds between emotional moves (was 6.0)

    # Gen20: enhanced per-LED mood lighting
    mood_led_wave_rate: float = 2.0                   # speed of LED wave animation
    mood_led_breath_rate: float = 1.5                 # breathing rate for calm moods
    mood_led_pulse_intensity: float = 0.4             # intensity of pulse overlay
    mood_led_gradient_blend: float = 0.3              # how much gradient blends between LEDs
    mood_celebration_rainbow_speed: float = 5.0       # rainbow speed during celebration
    mood_stressed_flicker_rate: float = 8.0           # flicker rate when stressed
    mood_exploring_hue_speed: float = 0.8             # hue rotation speed when exploring
    mood_confident_glow_rate: float = 1.0             # glow pulse rate when confident

    # Gen21: enhanced straight-line cruise mode
    cruise_mode_activation_distance: float = 65.0     # front distance to enter cruise mode
    cruise_mode_min_duration: float = 1.5             # minimum seconds to stay in cruise mode
    cruise_mode_heading_lock_strength: float = 0.85   # how strongly to resist turns (0-1)
    cruise_mode_speed_boost: float = 0.08             # extra speed in cruise mode
    cruise_mode_turn_resistance: float = 12.0         # score penalty per 10° of turn in cruise
    cruise_mode_max_duration: float = 8.0             # max seconds before cruise mode exits
    cruise_mode_graceful_exit_distance: float = 40.0  # front distance to gracefully exit cruise

    # Gen21: smooth arc steering
    arc_steering_max_rate: float = 0.15               # max steering change per loop cycle
    arc_steering_accel: float = 0.04                  # steering acceleration when changing direction
    arc_steering_decel: float = 0.08                  # steering deceleration when centering

    # Gen21: emotional moves Gen21
    satisfied_purr_duration: float = 0.8              # duration of satisfied purr
    determined_lunge_duration: float = 0.6            # duration of determined lunge
    grateful_bow_duration: float = 0.7                # duration of grateful bow
    victory_lap_duration: float = 1.5                 # duration of victory lap
    curious_sniff_duration: float = 0.9               # duration of curious sniff
    gentle_weave_duration: float = 1.2                # duration of gentle weave
    happy_hop_duration: float = 0.5                   # duration of happy hop
    emotional_move_cooldown: float = 4.0              # minimum seconds between emotional moves (was 5.0)

    # Gen22: boustrophedon coverage path planning
    boustrophedon_activation_coverage: float = 0.15     # grid coverage ratio below which boustrophedon activates
    boustrophedon_lane_width_cm: float = 25.0            # width of each coverage lane
    boustrophedon_duration_s: float = 6.0                # max duration of a boustrophedon run
    boustrophedon_speed: float = 0.50                    # speed during boustrophedon coverage
    boustrophedon_turn_rate: float = 0.35                # turn rate at lane end
    boustrophedon_cooldown_s: float = 20.0               # cooldown between boustrophedon runs
    boustrophedon_obstacle_margin: float = 18.0          # minimum clearance to continue boustrophedon

    # Gen22: room boundary detection
    room_boundary_window: int = 10                       # scans to confirm a boundary
    room_boundary_distance: float = 120.0                # distance threshold for boundary detection
    room_shape_cells: int = 16                           # angular sectors for room shape map
    room_shape_decay: float = 0.999                      # decay for room shape memory

    # Gen22: new emotional moves Gen22
    zoomies_duration: float = 1.2                        # duration of zoomies burst
    play_bow_duration: float = 0.8                       # duration of play bow
    peek_a_boo_duration: float = 0.6                     # duration of peek-a-boo
    stalk_mode_duration: float = 1.5                     # duration of stalk mode
    greeting_duration: float = 1.0                       # duration of greeting
    sleep_mode_duration: float = 2.0                     # duration of sleep mode
    backing_dance_duration: float = 0.8                  # duration of backing dance
    serpentine_duration: float = 1.5                     # duration of serpentine
    emotional_move_cooldown: float = 3.5                 # minimum seconds between emotional moves (was 4.0)

    # Gen22: enhanced emotional triggers
    zoomies_streak_threshold: int = 15                   # success streak for zoomies
    play_bow_open_space_threshold: float = 100.0         # front distance for play bow
    stalk_mode_distance: float = 35.0                    # front distance to trigger stalk mode
    greeting_distance_threshold: float = 20.0            # sudden close object triggers greeting
    sleep_mode_inactivity_s: float = 12.0                # seconds of low activity for sleep mode
    serpentine_coverage_threshold: float = 0.30          # grid coverage for serpentine celebration

    # Gen22: enhanced lighting
    mood_aurora_wave_speed: float = 0.3                  # speed of aurora wave across LEDs
    mood_rainbow_burst_duration: float = 3.0             # duration of rainbow burst effect
    mood_breathing_rate: float = 1.5                     # base breathing rate for calm states
    mood_strobe_rate: float = 15.0                       # strobe rate for excited states
    mood_gradient_blur: float = 0.4                      # how much colors blend between adjacent LEDs

    # Gen22: enhanced straight-line cruise enhancements
    cruise_mode_lane_keeping: float = 0.15               # lane-keeping steer correction in cruise
    cruise_mode_lookahead_distance: float = 45.0         # cm to look ahead in cruise mode
    cruise_mode_obstacle_anticipation: float = 0.25      # speed reduction factor for anticipated obstacles
    straight_line_momentum_gain: float = 0.08            # momentum build rate per second going straight
    straight_line_momentum_max: float = 1.0              # maximum straight-line momentum
    straight_line_momentum_decay: float = 0.95           # decay when turning
    straight_line_momentum_resistance: float = 15.0      # score penalty per 10° turn when momentum is high
    straight_line_momentum_speed_boost: float = 0.10     # speed boost at max momentum
    directional_inertia_threshold: float = 3.0           # seconds before directional inertia kicks in
    directional_inertia_strength: float = 0.20           # how strongly to resist heading changes

    # ─── Gen23: topological room graph ────────────────────────────────────
    topological_node_max: int = 30                       # max nodes in topological graph
    topological_node_spacing_cm: float = 80.0            # min distance between nodes
    topological_edge_expiry_s: float = 45.0              # seconds before edge expires
    topological_unvisited_bonus: float = 30.0            # scoring bonus for unvisited nodes
    topological_visited_penalty: float = -10.0           # penalty for recently visited nodes
    topological_frontier_bonus: float = 20.0             # bonus for nodes leading to unexplored areas
    topological_room_open_threshold: float = 90.0        # front distance to classify as "room" node
    topological_corridor_width_max: float = 60.0         # max side-to-side to classify as "corridor" node
    topological_doorway_width_min: float = 35.0          # min side opening to classify as "doorway"
    topological_node_decay: float = 0.9995               # per-loop decay for node salience

    # ─── Gen23: VFH-inspired polar histogram ──────────────────────────────
    vfh_sector_count: int = 36                           # 10-degree sectors for polar histogram
    vfh_threshold_distance: float = 25.0                 # distance below which sector is blocked
    vfh_valley_min_width: int = 2                        # min consecutive free sectors for a valley
    vfh_valley_center_bonus: float = 18.0                # bonus for heading toward valley center
    vfh_valley_width_bonus: float = 8.0                  # bonus per sector of valley width
    vfh_smoothness_bonus: float = 12.0                   # bonus for smooth heading transitions
    vfh_histogram_decay: float = 0.998                   # per-loop decay for polar histogram

    # ─── Gen23: enhanced exploration intelligence ─────────────────────────
    exploration_direction_history_size: int = 20         # recent directions to avoid repeating
    exploration_direction_repeat_penalty: float = -15.0  # penalty for repeating same direction
    exploration_area_classification_window: int = 6      # scans to classify current area type
    exploration_return_to_open_bonus: float = 22.0       # bonus for heading back toward known open space
    exploration_return_expiry_s: float = 30.0            # how long to remember open space beacons
    exploration_beacon_max: int = 8                      # max open space beacons to track

    # ─── Gen23: new emotional moves Gen23 ─────────────────────────────────
    explorer_pride_duration: float = 1.2                 # duration of explorer pride
    mapping_joy_duration: float = 1.5                    # duration of mapping joy
    corridor_dance_duration: float = 1.0                 # duration of corridor dance
    open_space_celebration_duration: float = 1.5         # duration of open space celebration
    wall_caress_duration: float = 1.2                    # duration of wall caress
    discovery_spin_duration: float = 0.8                 # duration of discovery spin
    emotional_move_cooldown: float = 3.0                 # minimum seconds between emotional moves (was 3.5)

    # ─── Gen23: enhanced LED wave system ──────────────────────────────────
    led_wave_speed: float = 2.5                          # speed of LED wave animation
    led_wave_intensity: float = 0.6                      # intensity of wave overlay
    led_wave_saturation: float = 0.85                    # saturation of wave colors
    led_breathing_rate: float = 1.2                      # base breathing rate for calm states
    led_pulse_on_streak: bool = True                     # pulse LEDs on success streaks
    led_pulse_streak_threshold: int = 10                 # streak threshold for LED pulse
    led_rainbow_on_celebration: bool = True              # rainbow wave during celebration moves

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
    last_push_time: float = field(default=-math.inf)
    danger_heatmap: Dict[int, float] = field(default_factory=dict)
    stress: float = field(default=0.0)
    pounce_heading: int = field(default=0)
    pounce_until: float = field(default=-math.inf)
    terrain_class: str = field(default="unknown")
    terrain_history: Deque[str] = field(init=False)
    curvature_history: Deque[float] = field(init=False)
    corner_detected: bool = field(default=False)
    corner_side: int = field(default=0)
    corner_detection_count: int = field(default=0)
    gap_width_cache: Dict[int, float] = field(default_factory=dict)
    obstacle_density_history: Deque[float] = field(init=False)
    path_confidence: float = field(default=1.0)
    path_confidence_history: Deque[float] = field(init=False)
    heading_momentum: float = field(default=0.0)
    drift_history: Deque[int] = field(init=False)
    lookahead_scores: Dict[int, float] = field(default_factory=dict)
    personality_boldness: float = field(default=0.5)  # 0=cautious, 1=bold
    personality_curiosity: float = field(default=0.5)  # 0=routine, 1=explorer
    personality_playfulness: float = field(default=0.5)  # 0=serious, 1=playful
    success_streak: int = field(default=0)
    recent_caution_ratio: float = field(default=0.0)
    caution_window: Deque[bool] = field(init=False)
    flow_memory: Deque[Tuple[int, float]] = field(init=False)  # (heading, quality) tuples
    danger_zone_memory: Dict[int, float] = field(default_factory=dict)
    recent_quips: Deque[str] = field(init=False)
    last_quip_time: float = field(default=-math.inf)
    escape_count: int = field(default=0)
    dead_end_count: int = field(default=0)
    push_count: int = field(default=0)
    pounce_count: int = field(default=0)
    total_loops: int = field(default=0)
    prev_front_distance: float = field(default=0.0)
    side_closing_history: Deque[float] = field(init=False)
    last_side_closing: float = field(default=0.0)
    escape_outcomes: Dict[int, float] = field(default_factory=dict)  # heading -> cumulative outcome score
    last_escape_heading: int = field(default=0)
    last_escape_front_after: float = field(default=0.0)
    persistence_map: Dict[int, float] = field(default_factory=dict)  # angle -> persistence score (how often blocked)
    persistence_history: Deque[Dict[int, float]] = field(init=False)  # recent scan snapshots for persistence tracking
    corridor_exit_detected: bool = field(default=False)
    corridor_exit_heading: int = field(default=0)
    corridor_exit_time: float = field(default=-math.inf)
    escape_direction_outcomes: Dict[str, float] = field(default_factory=dict)  # "left"/"right" -> outcome
    escape_direction_counts: Dict[str, int] = field(default_factory=dict)  # "left"/"right" -> count
    last_notable_event: str = field(default="")
    last_notable_event_time: float = field(default=-math.inf)
    event_memory: Deque[str] = field(init=False)
    longest_open_streak: float = field(default=0.0)
    total_escapes_ever: int = field(default=0)
    recovery_stage: int = field(default=0)  # progressive recovery: 0=normal, 1=wiggling, 2=reverse-spin, 3=full-spin
    recovery_stage_time: float = field(default=-math.inf)
    dead_end_prediction_count: int = field(default=0)  # consecutive scans suggesting dead-end forming
    dead_end_predicted: bool = field(default=False)
    dead_end_prediction_time: float = field(default=-math.inf)
    obstacle_shape: str = field(default="unknown")  # inferred shape: wall/corner/isolated/unknown
    obstacle_shape_history: Deque[str] = field(init=False)
    success_map: Dict[int, float] = field(default_factory=dict)  # cell -> success quality
    terrain_memory: Dict[int, float] = field(default_factory=dict)  # cell -> openness quality
    last_achievement: str = field(default="")
    last_achievement_time: float = field(default=-math.inf)
    achievement_count: int = field(default=0)
    self_aware_quip_history: Deque[str] = field(init=False)
    last_self_aware_quip_time: float = field(default=-math.inf)
    navigation_insights: int = field(default=0)  # how many times the bot has "learned" something
    tactical_patience_active: bool = field(default=False)
    patience_start_time: float = field(default=-math.inf)
    patience_reassess_count: int = field(default=0)
    junction_detected: bool = field(default=False)
    junction_type: str = field(default="")  # "T", "cross", "open"
    junction_time: float = field(default=-math.inf)
    junction_exit_headings: list = field(default_factory=list)
    near_miss_history: Deque[float] = field(init=False)
    near_miss_angles: Deque[int] = field(init=False)
    strategy_reward: Dict[str, float] = field(default_factory=dict)
    strategy_eval_count: int = field(default=0)
    last_strategy_change_time: float = field(default=0.0)
    last_random_exploration_time: float = field(default=0.0)
    exploration_direction_bias: float = field(default=0.0)
    current_exploration_strategy: str = field(default="balanced")
    loop_detection_positions: Deque[Tuple[int, float]] = field(init=False)
    loop_count: int = field(default=0)
    last_loop_detection_time: float = field(default=-math.inf)
    wall_following_escape_active: bool = field(default=False)
    wall_following_escape_start: float = field(default=-math.inf)
    wall_following_escape_direction: int = field(default=1)
    anticipatory_steer_history: Deque[Tuple[int, float]] = field(init=False)
    predicted_openings: Dict[int, float] = field(default_factory=dict)
    last_prediction_time: float = field(default=-math.inf)
    heading_hysteresis_count: int = field(default=0)
    heading_hysteresis_last: int = field(default=0)
    obstacle_velocity_map: Dict[int, Deque[Tuple[float, float]]] = field(default_factory=dict)
    moving_obstacles: set = field(default_factory=set)
    environmental_memory: Deque[Tuple[float, float, str]] = field(init=False)
    last_narrative_time: float = field(default=-math.inf)
    last_commentary_time: float = field(default=-math.inf)
    recent_narrative_topics: Deque[str] = field(init=False)
    side_corridor_detected: bool = field(default=False)
    side_corridor_heading: int = field(default=0)
    side_corridor_time: float = field(default=-math.inf)
    scoring_term_weights: Dict[str, float] = field(default_factory=dict)
    scoring_term_rewards: Dict[str, Deque[float]] = field(default_factory=dict)
    obstacle_cluster_detected: bool = field(default=False)
    obstacle_cluster_tightness: float = field(default=0.0)
    obstacle_cluster_history: Deque[float] = field(init=False)
    flow_state: str = field(default="normal")  # "flow", "normal", "struggle"
    flow_state_ratio: float = field(default=0.5)
    flow_state_history: Deque[bool] = field(init=False)
    last_fortune_time: float = field(default=-math.inf)
    fortune_predictions: Deque[str] = field(init=False)

    # Gen18: coverage map state
    coverage_map: Dict[int, float] = field(default_factory=dict)  # cell -> visit frequency
    coverage_map_last_cell: int = field(default=0)
    total_coverage_cells_visited: int = field(default=0)

    # Gen18: frontier state
    frontier_map_gen18: Dict[int, float] = field(default_factory=dict)  # cell -> frontier score

    # Gen18: straight-line state
    straight_line_start_time: float = field(default=-math.inf)
    straight_line_heading: int = field(default=0)
    directional_persistence: float = field(default=0.0)
    straight_line_bonus_active: bool = field(default=False)

    # Gen18: emotional move state
    emotional_move_active: bool = field(default=False)
    emotional_move_start: float = field(default=-math.inf)
    emotional_move_type: str = field(default="")
    last_emotional_move: float = field(default=-math.inf)
    emotional_move_step: int = field(default=0)

    # Gen18: mood state
    current_mood: str = field(default="neutral")  # neutral/celebration/stressed/exploring/confident
    mood_start_time: float = field(default=-math.inf)
    celebration_trigger: str = field(default="")
    mood_hue_offset: float = field(default=0.0)

    # Gen19: 2D grid exploration map
    exploration_grid: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (x,y) -> visit frequency
    current_grid_cell: Tuple[int, int] = field(default=(0, 0))
    grid_frontier_cells: set = field(default_factory=set)  # cells adjacent to visited but unvisited
    estimated_x: float = field(default=0.0)  # estimated X position (dead reckoning)
    estimated_y: float = field(default=0.0)  # estimated Y position (dead reckoning)
    estimated_heading_deg: float = field(default=0.0)  # estimated heading in degrees
    total_grid_cells_visited: int = field(default=0)
    last_grid_frontier_update: float = field(default=-math.inf)

    # Gen19: spiral exploration state
    spiral_active: bool = field(default=False)
    spiral_start_time: float = field(default=-math.inf)
    spiral_angle: float = field(default=0.0)
    last_spiral_time: float = field(default=-math.inf)

    # Gen19: path memory
    path_heading_history: Deque[int] = field(init=False)
    path_smoothness_score: float = field(default=0.0)

    # Gen19: discovery excitement
    last_discovery_time: float = field(default=-math.inf)
    discovery_count: int = field(default=0)
    largest_open_space_seen: float = field(default=0.0)
    discovery_light_end: float = field(default=-math.inf)

    # Gen19: enhanced mood lighting
    enhanced_mood_effect: str = field(default="")  # aurora/heartbeat/rainbow_wave
    enhanced_mood_start: float = field(default=-math.inf)

    # Gen20: waypoint/landmark memory
    landmark_memory: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)  # id -> (x, y, salience)
    landmark_visit_count: Dict[int, int] = field(default_factory=dict)  # id -> visit count
    next_landmark_id: int = field(default=0)
    last_landmark_registration: float = field(default=-math.inf)

    # Gen20: enhanced straight-line state
    straight_lock_start_time: float = field(default=-math.inf)
    straight_lock_heading: int = field(default=0)
    straight_lock_active: bool = field(default=False)
    last_non_zero_heading: int = field(default=0)
    straight_line_recovery_active: bool = field(default=False)

    # Gen20: directed frontier exploration
    last_frontier_target_time: float = field(default=-math.inf)
    frontier_target_heading: int = field(default=0)
    frontier_directed_active: bool = field(default=False)

    # Gen20: enhanced per-LED mood state
    led_animation_phase: float = field(default=0.0)
    last_mood_lighting_update: float = field(default=0.0)

    # Gen21: enhanced straight-line cruise mode
    cruise_mode_active: bool = field(default=False)
    cruise_mode_start_time: float = field(default=-math.inf)
    cruise_mode_distance_traveled: float = field(default=0.0)
    cruise_mode_heading: int = field(default=0)
    cruise_mode_escalation: int = field(default=0)

    # Gen21: smooth arc steering
    arc_steering_target: float = field(default=0.0)
    arc_steering_current: float = field(default=0.0)
    arc_steering_rate: float = field(default=0.0)

    # Gen21: dead-zone awareness
    dead_zone_map: Dict[Tuple[int, int], float] = field(default_factory=dict)
    last_dead_zone_update: float = field(default=-math.inf)

    # Gen21: enhanced emotional state
    satisfaction_level: float = field(default=0.0)
    curiosity_drive: float = field(default=0.0)
    last_satisfaction_trigger: float = field(default=-math.inf)

    # Gen21: dynamic lighting state
    light_show_active: bool = field(default=False)
    light_show_type: str = field(default="")
    light_show_start: float = field(default=-math.inf)
    led_wave_phase: float = field(default=0.0)

    # Gen22: boustrophedon coverage path planning
    boustrophedon_active: bool = field(default=False)
    boustrophedon_start_time: float = field(default=-math.inf)
    boustrophedon_lane_direction: int = field(default=1)  # 1=forward, -1=backward
    boustrophedon_current_lane: int = field(default=0)
    boustrophedon_last_run_time: float = field(default=-math.inf)
    boustrophedon_lane_angle: float = field(default=0.0)

    # Gen22: room boundary detection
    room_shape_map: Dict[int, float] = field(default_factory=dict)
    room_boundary_confirmed: bool = field(default=False)
    last_boundary_update: float = field(default=-math.inf)
    boundary_scan_history: Deque[Dict[int, float]] = field(init=False)

    # Gen22: enhanced emotional state tracking
    playfulness_level: float = field(default=0.0)
    energy_level: float = field(default=1.0)
    last_zoomies_time: float = field(default=-math.inf)
    last_greeting_time: float = field(default=-math.inf)
    last_sleep_mode_time: float = field(default=-math.inf)
    low_activity_start: float = field(default=-math.inf)

    # Gen22: straight-line momentum
    straight_line_momentum: float = field(default=0.0)
    straight_line_momentum_heading: int = field(default=0)
    straight_line_momentum_start: float = field(default=-math.inf)

    # Gen23: topological room graph
    topological_nodes: Dict[int, Tuple[float, float, str, float]] = field(default_factory=dict)  # id -> (x, y, type, salience)
    topological_edges: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (from,to) -> last_used_time
    topological_visited: set = field(default_factory=set)  # node ids that have been visited
    topological_current_node: int = field(default=-1)
    next_topological_id: int = field(default=0)
    last_topological_registration: float = field(default=-math.inf)
    topological_area_history: Deque[str] = field(init=False)

    # Gen23: VFH polar histogram
    vfh_histogram: Dict[int, float] = field(default_factory=dict)  # sector -> obstacle density
    vfh_valleys: list = field(default_factory=list)  # [(start_sector, end_sector, center_angle, width)]
    last_vfh_update: float = field(default=-math.inf)

    # Gen23: exploration intelligence
    exploration_direction_history: Deque[int] = field(init=False)
    open_space_beacons: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)  # id -> (heading, x, y)
    next_beacon_id: int = field(default=0)
    last_area_classification: str = field(default="unknown")
    area_classification_history: Deque[str] = field(init=False)

    def __post_init__(self):
        self.front_history = deque(maxlen=self.config.front_history_size)
        self.recent_turns = deque(maxlen=self.config.recent_turn_memory)
        self.stuck_escape_times = deque(maxlen=self.config.stuck_escape_count + 1)
        self.front_rate_history = deque(maxlen=self.config.approach_rate_window)
        self.angle_rate_history = {}
        self.complexity_history = deque(maxlen=self.config.environment_complexity_window)
        self.heading_history = deque(maxlen=self.config.oscillation_window)
        self.scan_confidence_history = deque(maxlen=self.config.scan_confidence_window)
        self.terrain_history = deque(maxlen=self.config.terrain_classify_window)
        self.curvature_history = deque(maxlen=self.config.curvature_window)
        self.obstacle_density_history = deque(maxlen=self.config.obstacle_density_window)
        self.path_confidence_history = deque(maxlen=self.config.path_confidence_window)
        self.drift_history = deque(maxlen=self.config.drift_correction_window)
        self.caution_window = deque(maxlen=20)
        self.flow_memory = deque(maxlen=self.config.flow_memory_window)
        self.recent_quips = deque(maxlen=self.config.quip_history_size)
        self.side_closing_history = deque(maxlen=self.config.side_closing_window)
        self.persistence_history = deque(maxlen=self.config.persistence_window)
        self.event_memory = deque(maxlen=20)
        self.obstacle_shape_history = deque(maxlen=self.config.obstacle_shape_window)
        self.self_aware_quip_history = deque(maxlen=8)
        self.near_miss_history = deque(maxlen=self.config.near_miss_window)
        self.near_miss_angles = deque(maxlen=self.config.near_miss_window)
        self.loop_detection_positions = deque(maxlen=30)
        self.anticipatory_steer_history = deque(maxlen=10)
        self.environmental_memory = deque(maxlen=self.config.environmental_memory_size)
        self.recent_narrative_topics = deque(maxlen=8)
        self.obstacle_cluster_history = deque(maxlen=self.config.obstacle_cluster_window)
        self.flow_state_history = deque(maxlen=self.config.flow_state_window)
        self.fortune_predictions = deque(maxlen=6)
        self.path_heading_history = deque(maxlen=self.config.path_memory_window)
        self.boundary_scan_history = deque(maxlen=self.config.room_boundary_window)
        self.topological_area_history = deque(maxlen=8)
        self.exploration_direction_history = deque(maxlen=self.config.exploration_direction_history_size)
        self.area_classification_history = deque(maxlen=self.config.exploration_area_classification_window)
        self._init_scoring_weights()
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
        if front_distance <= self.config.adaptive_scan_urgency_distance:
            urgent_interval = self.config.adaptive_scan_base * self.config.adaptive_scan_urgency_multiplier
            if now - self.last_scan_time >= urgent_interval:
                return True
        positive_scan = [v for v in self.last_scan.values() if v > 0.0]
        was_open = bool(positive_scan) and all(v > self.config.open_space_distance for v in positive_scan)
        base_interval = self.config.open_space_scan_s if was_open else self.get_adaptive_scan_interval()
        if self.obstacle_density_history:
            avg_density = sum(self.obstacle_density_history) / len(self.obstacle_density_history)
            if avg_density > self.config.obstacle_density_threshold:
                base_interval *= self.config.density_scan_interval_multiplier
        if self.obstacle_cluster_detected:
            base_interval *= self.config.cluster_scan_multiplier
        if self.flow_state == "flow":
            base_interval *= self.config.flow_state_scan_extension
        elif self.flow_state == "struggle":
            base_interval *= self.config.struggle_scan_urgency
        proactive_interval = base_interval
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
        self.update_terrain(self.last_scan)
        self.update_obstacle_density(self.last_scan)
        self.update_path_confidence(front_distance, self.last_scan)
        self.update_persistence(self.last_scan)
        self.update_corridor_exit(front_distance, self.last_scan, now)
        self.gap_width_cache.clear()

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
        if not self.exploration_memory:
            return 0.0

        spread = max(1.0, self.config.exploration_spread_degrees)
        penalty = 0.0
        for past_angle, visits in self.exploration_memory.items():
            angle_gap = abs(angle - past_angle)
            if angle_gap > spread:
                continue
            proximity = 1.0 - (angle_gap / spread)
            penalty += visits * proximity
        return -penalty * self.config.exploration_penalty_scale

    def decay_exploration_memory(self):
        decay = self.config.exploration_decay
        for angle in list(self.exploration_memory.keys()):
            self.exploration_memory[angle] *= decay
            if self.exploration_memory[angle] < 0.05:
                del self.exploration_memory[angle]

    def record_heading_choice(self, angle: int):
        if angle == 0:
            return
        self.exploration_memory[angle] = self.exploration_memory.get(angle, 0.0) + 1.0

    def add_stress(self, amount: float):
        self.stress = min(1.0, self.stress + amount)

    def decay_stress(self):
        self.stress *= self.config.stress_decay
        if self.stress < 0.01:
            self.stress = 0.0

    def update_personality(self, front_distance: float, mode: str):
        """Adapt personality based on recent experience."""
        was_caution = front_distance <= self.config.caution_distance
        self.caution_window.append(was_caution)
        self.recent_caution_ratio = sum(self.caution_window) / len(self.caution_window)

        if mode == "drive" and front_distance > self.config.cruise_distance:
            self.success_streak += 1
        elif mode in ("escape",):
            self.success_streak = 0

        boldness_target = 0.5
        boldness_target += (0.3 * self.success_streak / max(1, self.success_streak + 5))
        boldness_target -= 0.3 * self.recent_caution_ratio
        boldness_target -= 0.2 * self.stress
        self.personality_boldness += self.config.personality_adapt_rate * (boldness_target - self.personality_boldness)
        self.personality_boldness = max(0.0, min(1.0, self.personality_boldness))

        curiosity_target = 0.5
        curiosity_target += 0.3 * (1.0 - self.recent_caution_ratio)
        curiosity_target -= 0.2 * self.stress
        self.personality_curiosity += self.config.personality_adapt_rate * (curiosity_target - self.personality_curiosity)
        self.personality_curiosity = max(0.0, min(1.0, self.personality_curiosity))

        playfulness_target = 0.5
        playfulness_target += 0.2 * (self.success_streak / max(1, self.success_streak + 8))
        playfulness_target -= 0.4 * self.stress
        self.personality_playfulness += self.config.personality_adapt_rate * (playfulness_target - self.personality_playfulness)
        self.personality_playfulness = max(0.0, min(1.0, self.personality_playfulness))

    def get_personality_label(self) -> str:
        """Return a short label describing current personality blend."""
        if self.stress > 0.6:
            return "panicked"
        if self.personality_boldness > 0.7:
            return "bold"
        if self.personality_boldness < 0.3:
            return "cautious"
        if self.personality_playfulness > 0.7:
            return "playful"
        if self.personality_curiosity > 0.7:
            return "curious"
        return "balanced"

    def plan_follow(self, front_distance: float) -> MotionCommand:
        """Maintain follow_target_distance from the object directly ahead."""
        front_distance = self.sanitize_distance(front_distance)
        target = self.config.follow_target_distance
        deadband = self.config.follow_deadband

        if front_distance <= 0.0:
            return MotionCommand(mode="follow_search", left_speed=0.0, right_speed=0.0,
                                 heading=0, colour=(0, 0, 60))

        error = front_distance - target
        if abs(error) <= deadband:
            speed = 0.0
        elif error > 0:
            speed = min(self.config.follow_max_speed, error * self.config.follow_kp)
        else:
            speed = -self.config.follow_reverse_speed

        return MotionCommand(mode="follow", left_speed=speed, right_speed=speed,
                             heading=0, colour=(0, 0, 0))

    def clear_pounce_commitment(self):
        self.pounce_heading = 0
        self.pounce_until = -math.inf

    def should_trigger_pounce(self, front_distance: float, heading: int) -> bool:
        front_distance = self.sanitize_distance(front_distance)
        if front_distance <= self.config.danger_distance:
            return False
        if front_distance > self.config.pounce_trigger_distance:
            return False
        if abs(heading) < self.config.pounce_min_heading:
            return False

        target_distance = self.sanitize_distance(self.last_scan.get(heading, 0.0))
        if target_distance <= 0.0:
            return False

        if target_distance - front_distance < self.config.pounce_min_clearance_gain:
            return False

        opposite_angle = -45 if heading > 0 else 45
        opposite_distance = self.sanitize_distance(self.last_scan.get(opposite_angle, 0.0))
        if opposite_distance > 0.0 and target_distance <= opposite_distance + 10.0:
            return False

        return True

    def build_pounce_command(self, heading: int, front_distance: float, now: float) -> MotionCommand:
        max_angle = max(abs(angle) for angle in self.config.scan_angles) or 1
        steer = max(-1.0, min(1.0, heading / max_angle))
        steer *= self.config.pounce_steer_gain
        steer = max(-1.0, min(1.0, steer))
        speed = self.config.pounce_speed
        self.current_speed = speed
        self.current_heading = heading
        self.path_commitment_heading = heading
        self.path_commitment_time = now
        self.record_heading_choice(heading)
        self.record_distance(speed, self.config.loop_delay_s)

        left_speed = max(-1.0, min(1.0, speed * (1.0 + steer * self.config.steer_gain)))
        right_speed = max(-1.0, min(1.0, speed * (1.0 - steer * self.config.steer_gain)))
        return MotionCommand(
            mode="peek_pounce",
            left_speed=left_speed,
            right_speed=right_speed,
            heading=heading,
            colour=self.motion_colour_for("peek_pounce", heading, speed, front_distance),
        )

    def record_danger(self, angles: list, score: float):
        """Burn a danger score into the heatmap for each given angle."""
        for angle in angles:
            self.danger_heatmap[angle] = min(1.0, self.danger_heatmap.get(angle, 0.0) + score)

    def decay_danger_heatmap(self):
        decay = self.config.danger_heatmap_decay
        for angle in list(self.danger_heatmap.keys()):
            self.danger_heatmap[angle] *= decay
            if self.danger_heatmap[angle] < 0.02:
                del self.danger_heatmap[angle]

    def format_heatmap(self) -> str:
        """Single-line ASCII bar showing danger level per scan angle."""
        blocks = ' ░▒▓█'
        parts = []
        for angle in self.config.scan_angles:
            d = self.danger_heatmap.get(angle, 0.0)
            ch = blocks[min(4, int(d * 5))]
            sign = '+' if angle > 0 else ''
            parts.append(f"{sign}{angle}:{ch}")
        return '  '.join(parts)

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

    def is_pushable_obstacle(self, front_distance: float) -> bool:
        """Return True when a single object blocks the front but sides are open — worth a brave push."""
        if front_distance <= self.config.danger_distance:
            return False
        if front_distance > self.config.dead_end_front_threshold:
            return False
        left = self.sanitize_distance(self.last_scan.get(-45, 0.0))
        right = self.sanitize_distance(self.last_scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return False
        return left > self.config.push_min_side and right > self.config.push_min_side

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
            turn_dir = -1 if self.path_commitment_heading >= 0 else 1
            return turn_dir * 80
        return heading

    def select_smart_escape_heading(self, scan: Dict[int, float], now: float) -> int:
        """Choose escape direction using full spatial reasoning, not just turn-habit memory."""
        ordered = tuple(sorted((a, self.sanitize_distance(d)) for a, d in scan.items()))
        candidates = []
        for idx, (angle, dist) in enumerate(ordered):
            if dist <= 0.0:
                continue
            support, _ = self.corridor_support(ordered, idx)
            score = dist * 0.3 + support * 0.7
            score += self.frontier_bonus_for_angle(angle)
            danger = self.danger_heatmap.get(angle, 0.0)
            score -= danger * self.config.danger_penalty_scale
            cell = self.angle_to_cell(angle)
            spatial = self.spatial_memory.get(cell, 0.0)
            score -= spatial * 15.0
            score += abs(angle) * self.config.escape_angle_bias
            asym = self.side_asymmetry_bias(angle, ordered, dist)
            score += asym
            candidates.append((angle, score))
        if not candidates:
            return -80 if sum(self.recent_turns) >= 0 else 80
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

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

    def clamp_colour_channel(self, value: float) -> int:
        return max(0, min(255, int(round(value))))

    def motion_colour_for(self, mode: str, heading: int, speed: float, front_distance: float) -> Colour:
        if mode == "escape":
            return (255, 48, 0)
        if mode == "dead_end_recovery":
            return (255, 128, 0)
        if mode == "peek_pounce":
            return (255, 72, 0)
        if mode != "drive":
            return (32, 32, 40)

        max_angle = max(abs(angle) for angle in self.config.scan_angles) or 1
        normalized_heading = max(-1.0, min(1.0, heading / max_angle))
        speed_ratio = max(0.0, min(1.0, speed / max(self.config.open_space_speed, 0.01)))
        if front_distance > 0.0:
            openness = max(
                0.0,
                min(
                    1.0,
                    (front_distance - self.config.caution_distance)
                    / max(1.0, self.config.open_space_distance - self.config.caution_distance),
                ),
            )
        else:
            openness = 0.0
        scan_roughness = 0.0
        if self.scan_confidence_history:
            scan_roughness = sum(self.scan_confidence_history) / len(self.scan_confidence_history)
        certainty = max(0.0, min(1.0, 1.0 - scan_roughness))

        if front_distance <= self.config.danger_distance:
            hue = 0.02
            saturation = 1.0
            value = 1.0
        elif front_distance <= self.config.caution_distance:
            hue = 0.08
            saturation = 1.0
            value = 0.65 + 0.25 * speed_ratio
        else:
            if normalized_heading < -0.15:
                warm_mix = min(1.0, (-normalized_heading - 0.15) / 0.85)
                hue = 0.03 + 0.06 * warm_mix
            elif normalized_heading > 0.15:
                cool_mix = min(1.0, (normalized_heading - 0.15) / 0.85)
                hue = 0.54 + 0.08 * cool_mix
            else:
                hue = 0.30 + normalized_heading * 0.08
            saturation = 0.70 + 0.20 * speed_ratio + 0.05 * certainty
            value = 0.45 + 0.25 * speed_ratio + 0.20 * openness + 0.10 * certainty

        hue = max(0.0, min(1.0, hue))
        saturation = max(0.0, min(1.0, saturation))
        value = max(0.0, min(1.0, value))
        red, green, blue = hsv_to_rgb(hue, saturation, value)
        return (
            self.clamp_colour_channel(red * 255.0),
            self.clamp_colour_channel(green * 255.0),
            self.clamp_colour_channel(blue * 255.0),
        )

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
            score -= spatial_obstacle * 15.0 * self.get_scoring_weight("spatial_memory")

        frontier_raw = self.frontier_bonus_for_angle(angle)
        score += frontier_raw * self.get_scoring_weight("frontier")

        danger = self.danger_heatmap.get(angle, 0.0)
        if danger > 0.02:
            score -= danger * self.config.danger_penalty_scale * self.get_scoring_weight("danger_heatmap")

        if self.path_commitment_heading != 0:
            angle_gap = abs(angle - self.path_commitment_heading)
            if angle_gap < 25:
                score += self.config.path_commitment_bonus

        score += self.flow_bonus_for_angle(angle) * self.get_scoring_weight("flow_memory")

        score += self.danger_zone_penalty_for_angle(angle) * self.get_scoring_weight("danger_zone")

        scan_dict = {a: d for a, d in ordered_scan}
        score += self.compute_option_value(angle, scan_dict) * 1.5 * self.get_scoring_weight("option_value")

        score += self.compute_novelty_bonus(angle) * self.get_scoring_weight("novelty")

        if self.escape_outcomes:
            w = self.get_scoring_weight("escape_outcome")
            for eh, outcome in self.escape_outcomes.items():
                if abs(eh - angle) < 25 and outcome > 0:
                    score += outcome * 12.0 * w

        score += self.side_asymmetry_bias(angle, ordered_scan, front_distance) * self.get_scoring_weight("side_asymmetry")

        score += self.persistence_penalty_for_angle(angle) * self.get_scoring_weight("persistence")

        score += self.escape_direction_bias(angle) * self.get_scoring_weight("escape_direction")

        if now is not None:
            score += self.corridor_exit_bonus_for_angle(angle, now) * self.get_scoring_weight("corridor_exit")

        score += self.success_bonus_for_angle(angle) * self.get_scoring_weight("success_map")

        score += self.terrain_memory_bonus_for_angle(angle) * self.get_scoring_weight("terrain_memory")

        if now is not None:
            score += self.junction_bonus_for_angle(angle, now) * self.get_scoring_weight("junction")

        score += self.near_miss_penalty_for_angle(angle) * self.get_scoring_weight("near_miss")

        score += self.get_loop_escape_bonus(angle, now) * self.get_scoring_weight("loop_escape") if now is not None else 0.0

        score += self.anticipatory_opening_bonus(angle) * self.get_scoring_weight("anticipatory")

        if now is not None:
            score += self.side_corridor_bonus_for_angle(angle, now) * self.get_scoring_weight("side_corridor")

        if self.is_moving_obstacle(angle):
            score -= 25.0 * self.get_scoring_weight("moving_obstacle")

        score += self.coverage_bonus_for_angle(angle) * self.get_scoring_weight("coverage")

        score += self.frontier_bonus_for_angle_gen18(angle) * self.get_scoring_weight("frontier_gen18")

        if now is not None:
            score += self.straight_line_score_bonus(angle, now) * self.get_scoring_weight("straight_line")

        score += self.path_smoothness_bonus_for_angle(angle) * self.get_scoring_weight("path_smoothness")

        score += self.grid_novelty_bonus_for_angle(angle) * self.get_scoring_weight("grid_novelty")

        score += self.grid_frontier_bonus_for_angle(angle) * self.get_scoring_weight("grid_frontier")

        # Gen20: landmark bonus
        score += self.landmark_bonus_for_angle(angle) * self.get_scoring_weight("landmark")

        # Gen20: straight lock bonus
        if now is not None:
            score += self.straight_lock_bonus(angle, now) * self.get_scoring_weight("straight_lock")

        # Gen20: directed frontier bonus
        if now is not None:
            score += self.frontier_directed_bonus_for_angle(angle, now) * self.get_scoring_weight("frontier_directed")

        # Gen21: cruise mode turn resistance
        if self.cruise_mode_active and abs(angle) >= 15:
            score -= (abs(angle) / 10.0) * self.config.cruise_mode_turn_resistance

        # Gen21: dead-zone awareness bonus
        score += self.dead_zone_bonus_for_heading(angle) * self.get_scoring_weight("dead_zone")

        # Gen22: straight-line momentum bonus/penalty
        score += self.get_straight_line_momentum_bonus(angle) * self.get_scoring_weight("straight_momentum")

        # Gen23: topological room graph bonus
        if now is not None:
            score += self.topological_bonus_for_heading(angle) * self.get_scoring_weight("topological")

        # Gen23: VFH-inspired valley bonus
        score += self.vfh_bonus_for_heading(angle) * self.get_scoring_weight("vfh")

        # Gen23: exploration return bonus
        score += self.exploration_return_bonus_for_heading(angle) * self.get_scoring_weight("exploration_return")

        # Gen23: exploration direction repeat penalty
        score += self.exploration_direction_penalty(angle) * self.get_scoring_weight("exploration_direction")

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
        second_score = -math.inf

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

            if not escape and self.config.gap_interpolation:
                scan_dict = {a: d for a, d in ordered_scan}
                score = self.apply_gap_width_filter(angle, scan_dict, score)

            if score > best_score or (math.isclose(score, best_score) and abs(angle) < abs(best_angle)):
                second_score = best_score
                best_score = score
                best_angle = angle
            elif score > second_score:
                second_score = score

        if not escape:
            best_angle = self.apply_heading_hysteresis(best_angle, best_score, second_score)

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

    def simulate_lookahead(self, start_angle: int, start_distance: float, scan: Dict[int, float], steps: int) -> float:
        """Simulate future positions and score how safe/promising a heading is."""
        total_score = 0.0
        current_angle = start_angle
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        for step in range(steps):
            projected_dist = start_distance - step * self.config.lookahead_distance_per_step
            if projected_dist <= self.config.danger_distance + self.config.lookahead_obstacle_margin:
                return total_score - 100.0 * (steps - step)
            angle_penalty = abs(current_angle) * self.config.lookahead_angle_penalty * step
            scan_dist = self.sanitize_distance(scan.get(current_angle, 0.0))
            if scan_dist > 0.0:
                total_score += scan_dist * 0.5
            total_score -= angle_penalty
            if step + 1 < steps:
                adjacent_angles = [a for a in scan if abs(a - current_angle) <= max(45, max_angle)]
                if adjacent_angles:
                    current_angle = max(adjacent_angles, key=lambda a: self.sanitize_distance(scan.get(a, 0.0)))
        return total_score

    def _apply_lookahead_to_heading(self, heading: int, front_distance: float) -> int:
        """Use lookahead scores to nudge heading toward safer/clearer directions."""
        if not self.lookahead_scores:
            return heading
        current_score = self.lookahead_scores.get(heading, 0.0)
        candidates = []
        for angle, score in self.lookahead_scores.items():
            if score > current_score + 5.0:
                angle_diff = abs(angle - heading)
                if angle_diff <= 45:
                    candidates.append((angle, score))
        if not candidates:
            return heading
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_angle, best_score = candidates[0]
        blend = min(0.5, (best_score - current_score) / 100.0)
        return int(heading + (best_angle - heading) * blend)

    def classify_terrain(self, scan: Dict[int, float]) -> str:
        if not scan:
            return "unknown"
        distances = [self.sanitize_distance(d) for d in scan.values() if d > 0.0]
        if len(distances) < 2:
            return "unknown"
        avg_dist = sum(distances) / len(distances)
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        side_sum = left + right if left > 0 and right > 0 else 0
        if side_sum > 0 and side_sum < self.config.terrain_corridor_width_max and avg_dist > self.config.caution_distance:
            return "corridor"
        if avg_dist >= self.config.terrain_open_space_threshold:
            return "open"
        if avg_dist <= self.config.terrain_cluttered_threshold:
            return "cluttered"
        return "mixed"

    def update_terrain(self, scan: Dict[int, float]):
        terrain = self.classify_terrain(scan)
        self.terrain_history.append(terrain)
        if self.terrain_history:
            counts = {}
            for t in self.terrain_history:
                counts[t] = counts.get(t, 0) + 1
            self.terrain_class = max(counts, key=counts.get)

    def get_terrain_speed_multiplier(self) -> float:
        if self.terrain_class == "open":
            return self.config.terrain_open_speed_multiplier
        if self.terrain_class == "cluttered":
            return self.config.terrain_cluttered_speed_multiplier
        if self.terrain_class == "corridor":
            return self.config.terrain_corridor_speed_multiplier
        return 1.0

    def calculate_curvature(self) -> float:
        if len(self.heading_history) < 2:
            return 0.0
        history = list(self.heading_history)
        total_change = 0.0
        for i in range(1, len(history)):
            total_change += abs(history[i] - history[i - 1])
        return total_change / (len(history) - 1)

    def get_curvature_speed_multiplier(self) -> float:
        curvature = self.calculate_curvature()
        if curvature > self.config.curvature_sharp_threshold:
            return max(0.5, 1.0 - (curvature - self.config.curvature_sharp_threshold) * self.config.curvature_speed_gain / 100.0)
        return 1.0

    def detect_corner(self, front_distance: float, scan: Dict[int, float]) -> Tuple[bool, int]:
        if front_distance > self.config.corner_front_threshold:
            return False, 0
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return False, 0
        asymmetry = abs(left - right)
        if asymmetry < self.config.corner_side_asymmetry:
            return False, 0
        open_side = -45 if left > right else 45
        return True, open_side

    def update_corner_detection(self, front_distance: float, scan: Dict[int, float]):
        is_corner, side = self.detect_corner(front_distance, scan)
        if is_corner:
            self.corner_detection_count += 1
            if self.corner_detection_count >= self.config.corner_detection_window:
                self.corner_detected = True
                self.corner_side = side
        else:
            self.corner_detection_count = max(0, self.corner_detection_count - 1)
            if self.corner_detection_count == 0:
                self.corner_detected = False
                self.corner_side = 0

    def estimate_gap_width(self, angle: int, scan: Dict[int, float]) -> float:
        if angle in self.gap_width_cache:
            return self.gap_width_cache[angle]
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        front = self.sanitize_distance(scan.get(0, 0.0))
        if angle < 0:
            side_reading = left
            opposite = right
        elif angle > 0:
            side_reading = right
            opposite = left
        else:
            side_reading = (left + right) / 2.0
            opposite = front
        if side_reading <= 0.0:
            width = 0.0
        else:
            width = side_reading * self.config.gap_width_side_blend
            if opposite > 0.0:
                width += opposite * (1.0 - self.config.gap_width_side_blend)
        width = max(0.0, width - self.config.gap_width_safety_margin)
        self.gap_width_cache[angle] = width
        return width

    def calculate_obstacle_density(self, scan: Dict[int, float]) -> float:
        if not scan:
            return 0.0
        blocked = sum(1 for d in scan.values() if 0 < self.sanitize_distance(d) < self.config.caution_distance)
        return blocked / len(scan)

    def update_obstacle_density(self, scan: Dict[int, float]):
        density = self.calculate_obstacle_density(scan)
        self.obstacle_density_history.append(density)

    def get_density_speed_multiplier(self) -> float:
        if not self.obstacle_density_history:
            return 1.0
        avg_density = sum(self.obstacle_density_history) / len(self.obstacle_density_history)
        if avg_density > self.config.obstacle_density_threshold:
            return 1.0 - self.config.density_speed_reduction
        return 1.0

    def update_path_confidence(self, front_distance: float, scan: Dict[int, float]):
        positive = sum(1 for d in scan.values() if d > 0.0)
        total = len(scan) if scan else 1
        coverage = positive / total
        self.path_confidence = self.path_confidence * self.config.path_confidence_decay + coverage * 0.1
        self.path_confidence = max(0.0, min(1.0, self.path_confidence))
        self.path_confidence_history.append(self.path_confidence)

    def get_path_confidence_multiplier(self) -> float:
        if not self.path_confidence_history:
            return 1.0
        avg_confidence = sum(self.path_confidence_history) / len(self.path_confidence_history)
        if avg_confidence < 0.5:
            return 0.85
        return 1.0

    def update_momentum(self, heading: int):
        if abs(heading) >= self.config.momentum_turn_threshold:
            sign = 1 if heading > 0 else -1
            self.heading_momentum = self.heading_momentum * self.config.momentum_decay + sign * 0.1
        else:
            self.heading_momentum *= self.config.momentum_decay
        self.heading_momentum = max(-1.0, min(1.0, self.heading_momentum))

    def get_momentum_speed_bonus(self) -> float:
        if abs(self.heading_momentum) > 0.3:
            return self.config.momentum_speed_bonus * abs(self.heading_momentum)
        return 0.0

    def estimate_drift(self) -> float:
        if len(self.drift_history) < 2:
            return 0.0
        history = list(self.drift_history)
        return sum(history) / len(history)

    def apply_drift_correction(self, steer: float) -> float:
        drift = self.estimate_drift()
        if abs(drift) > self.config.drift_threshold:
            correction = -drift * self.config.drift_correction_gain
            return max(-1.0, min(1.0, steer + correction))
        return steer

    def update_flow_memory(self, heading: int, front_distance: float):
        improvement = front_distance - self.prev_front_distance
        if self.prev_front_distance > 0:
            quality = improvement / self.prev_front_distance
        else:
            quality = 0.0
        self.flow_memory.append((heading, quality))
        self.prev_front_distance = front_distance

    def flow_bonus_for_angle(self, angle: int) -> float:
        if len(self.flow_memory) < 2:
            return 0.0
        total_bonus = 0.0
        total_weight = 0.0
        for i, (h, q) in enumerate(self.flow_memory):
            if abs(h - angle) < 20:
                weight = (i + 1) / len(self.flow_memory)
                total_bonus += q * weight
                total_weight += weight
        if total_weight > 0:
            return (total_bonus / total_weight) * self.config.flow_memory_gain * 10.0
        return 0.0

    def decay_flow_memory(self):
        decay = self.config.flow_memory_decay
        new_memory = deque(maxlen=self.config.flow_memory_window)
        for h, q in self.flow_memory:
            new_memory.append((h, q * decay))
        self.flow_memory = new_memory

    def danger_zone_to_cell(self, angle: int) -> int:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        normalized = (angle + max_angle) / (2.0 * max_angle)
        cell = int(normalized * self.config.danger_zone_cells)
        return max(0, min(self.config.danger_zone_cells - 1, cell))

    def update_danger_zone(self, angles: list):
        for angle in angles:
            cell = self.danger_zone_to_cell(angle)
            self.danger_zone_memory[cell] = min(
                1.0,
                self.danger_zone_memory.get(cell, 0.0) + self.config.danger_zone_event_score
            )

    def decay_danger_zone(self):
        decay = self.config.danger_zone_decay
        for cell in list(self.danger_zone_memory.keys()):
            self.danger_zone_memory[cell] *= decay
            if self.danger_zone_memory[cell] < 0.02:
                del self.danger_zone_memory[cell]

    def danger_zone_penalty_for_angle(self, angle: int) -> float:
        cell = self.danger_zone_to_cell(angle)
        danger = self.danger_zone_memory.get(cell, 0.0)
        return -danger * self.config.danger_zone_penalty

    def update_persistence(self, scan: Dict[int, float]):
        """Track how persistently obstacles appear at each angle across scans.
        High persistence = static obstacle (wall). Low persistence = transient (person, movable object)."""
        snapshot = {}
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            is_blocked = 0.0 if dist > self.config.caution_distance else (1.0 - dist / self.config.caution_distance)
            snapshot[angle] = is_blocked
        self.persistence_history.append(snapshot)
        for angle in set().union(*[s.keys() for s in self.persistence_history] if self.persistence_history else []):
            values = [s.get(angle, 0.0) for s in self.persistence_history]
            self.persistence_map[angle] = sum(values) / len(values) if values else 0.0

    def persistence_penalty_for_angle(self, angle: int) -> float:
        """Extra penalty for angles with persistent obstacles — walls that never move."""
        persistence = self.persistence_map.get(angle, 0.0)
        if persistence > self.config.persistence_obstacle_threshold:
            return -persistence * self.config.persistence_static_penalty
        return 0.0

    def detect_corridor_exit(self, front_distance: float, scan: Dict[int, float]) -> Tuple[bool, int]:
        """Detect when a corridor is opening up ahead and identify the best exit heading."""
        if front_distance < self.config.corridor_exit_front_threshold:
            return False, 0
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left > self.config.corridor_exit_min_side and right > self.config.corridor_exit_min_side:
            best_exit = 0
            if left > right:
                best_exit = -45
            elif right > left:
                best_exit = 45
            return True, best_exit
        return False, 0

    def update_corridor_exit(self, front_distance: float, scan: Dict[int, float], now: float):
        """Track detected corridor exits with time-based memory."""
        detected, heading = self.detect_corridor_exit(front_distance, scan)
        if detected:
            self.corridor_exit_detected = True
            self.corridor_exit_heading = heading
            self.corridor_exit_time = now
        elif now - self.corridor_exit_time > self.config.corridor_exit_anticipation_s:
            self.corridor_exit_detected = False
            self.corridor_exit_heading = 0

    def corridor_exit_bonus_for_angle(self, angle: int, now: float) -> float:
        """Score bonus for heading toward a detected corridor exit."""
        if not self.corridor_exit_detected:
            return 0.0
        if now - self.corridor_exit_time > self.config.corridor_exit_anticipation_s:
            return 0.0
        if angle == self.corridor_exit_heading:
            return self.config.corridor_exit_bonus
        angle_gap = abs(angle - self.corridor_exit_heading)
        if angle_gap <= 45:
            return self.config.corridor_exit_bonus * (1.0 - angle_gap / 45.0)
        return 0.0

    def record_escape_direction_outcome(self, heading: int, outcome: float):
        """Track escape outcomes by general direction (left vs right) for learning."""
        direction = "left" if heading < 0 else "right"
        prev = self.escape_direction_outcomes.get(direction, 0.0)
        count = self.escape_direction_counts.get(direction, 0)
        lr = self.config.escape_direction_learning_rate
        self.escape_direction_outcomes[direction] = prev * (1.0 - lr) + outcome * lr
        self.escape_direction_counts[direction] = count + 1

    def escape_direction_bias(self, angle: int) -> float:
        """Bias toward directions with historically better escape outcomes."""
        direction = "left" if angle < 0 else "right"
        outcome = self.escape_direction_outcomes.get(direction, 0.0)
        count = self.escape_direction_counts.get(direction, 0)
        if count < 2:
            return 0.0
        return outcome * 8.0

    def predict_dead_end(self, front_distance: float, scan: Dict[int, float]) -> bool:
        """Proactively detect when a dead-end is forming before fully entering it."""
        if front_distance > self.config.dead_end_prediction_front_max:
            self.dead_end_prediction_count = 0
            self.dead_end_predicted = False
            return False
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            self.dead_end_prediction_count = 0
            return False
        if left < self.config.dead_end_prediction_side_max and right < self.config.dead_end_prediction_side_max:
            self.dead_end_prediction_count += 1
            if self.dead_end_prediction_count >= self.config.dead_end_prediction_window:
                self.dead_end_predicted = True
                self.dead_end_prediction_time = time.monotonic()
                return True
        else:
            self.dead_end_prediction_count = max(0, self.dead_end_prediction_count - 1)
        return self.dead_end_predicted

    def infer_obstacle_shape(self, front_distance: float, scan: Dict[int, float]) -> str:
        """Classify the obstacle ahead as wall/corner/isolated/unknown based on scan pattern."""
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if front_distance <= 0.0 or left <= 0.0 or right <= 0.0:
            return "unknown"
        if front_distance < self.config.shape_wall_front_threshold and abs(left - right) < self.config.shape_wall_side_tolerance:
            shape = "wall"
        elif front_distance < self.config.dead_end_front_threshold and abs(left - right) > self.config.shape_corner_side_gap:
            shape = "corner"
        elif front_distance < self.config.shape_isolated_max_width and left > self.config.cruise_distance and right > self.config.cruise_distance:
            shape = "isolated"
        else:
            shape = "unknown"
        self.obstacle_shape_history.append(shape)
        if self.obstacle_shape_history:
            counts = {}
            for s in self.obstacle_shape_history:
                counts[s] = counts.get(s, 0) + 1
            self.obstacle_shape = max(counts, key=counts.get)
        return self.obstacle_shape

    def success_cell_for_angle(self, angle: int) -> int:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        normalized = (angle + max_angle) / (2.0 * max_angle)
        cell = int(normalized * self.config.success_map_cells)
        return max(0, min(self.config.success_map_cells - 1, cell))

    def update_success_map(self, heading: int, front_before: float, front_after: float):
        """Record whether a heading led to improvement or deterioration."""
        if front_before <= 0.0:
            return
        quality = (front_after - front_before) / front_before
        cell = self.success_cell_for_angle(heading)
        prev = self.success_map.get(cell, 0.0)
        self.success_map[cell] = prev * 0.7 + quality * 0.3

    def decay_success_map(self):
        decay = self.config.success_map_decay
        for cell in list(self.success_map.keys()):
            self.success_map[cell] *= decay
            if abs(self.success_map[cell]) < 0.02:
                del self.success_map[cell]

    def success_bonus_for_angle(self, angle: int) -> float:
        """Reward headings toward directions that historically led to progress."""
        cell = self.success_cell_for_angle(angle)
        quality = self.success_map.get(cell, 0.0)
        if quality > self.config.success_map_quality_threshold:
            return quality * self.config.success_map_gain
        return 0.0

    def terrain_cell_for_angle(self, angle: int) -> int:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        normalized = (angle + max_angle) / (2.0 * max_angle)
        cell = int(normalized * self.config.terrain_memory_cells)
        return max(0, min(self.config.terrain_memory_cells - 1, cell))

    def update_terrain_memory(self, scan: Dict[int, float]):
        """Remember which directions led to open vs cluttered areas."""
        decay = self.config.terrain_memory_decay
        for cell in self.terrain_memory:
            self.terrain_memory[cell] *= decay
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            if dist <= 0.0:
                continue
            cell = self.terrain_cell_for_angle(angle)
            openness = min(1.0, dist / self.config.cruise_distance)
            prev = self.terrain_memory.get(cell, 0.5)
            self.terrain_memory[cell] = prev * 0.6 + openness * 0.4

    def terrain_memory_bonus_for_angle(self, angle: int) -> float:
        """Bonus for heading toward remembered open terrain."""
        cell = self.terrain_cell_for_angle(angle)
        openness = self.terrain_memory.get(cell, 0.5)
        if openness > 0.6:
            return (openness - 0.5) * self.config.terrain_memory_open_bonus
        return 0.0

    def should_trigger_patience(self, front_distance: float, scan: Dict[int, float], now: float) -> bool:
        """Detect ambiguous situations where pausing to reassess is better than rushing."""
        if self.tactical_patience_active:
            return False
        if front_distance > self.config.tactical_patience_front_max:
            return False
        if front_distance <= self.config.danger_distance:
            return False
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return False
        if left < self.config.tactical_patience_side_min and right < self.config.tactical_patience_side_min:
            return False
        front_variance = abs(left - right)
        if front_variance < 20.0:
            return False
        return True

    def activate_patience(self, now: float):
        self.tactical_patience_active = True
        self.patience_start_time = now
        self.patience_reassess_count += 1

    def reassess_after_patience(self, scan: Dict[int, float], front_distance: float, now: float) -> int:
        """After patience pause, choose heading with extra deliberation."""
        ordered = tuple(sorted((a, self.sanitize_distance(d)) for a, d in scan.items()))
        candidate_scan = self.build_gap_headings(ordered)
        heading = self._choose_best_heading(candidate_scan, front_distance, now=now, escape=False)
        self.tactical_patience_active = False
        return heading

    def detect_junction(self, front_distance: float, scan: Dict[int, float]) -> Tuple[bool, str, list]:
        """Detect junctions: T-junctions, cross-corridors, or open spaces."""
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return False, "", []

        exits = []
        if left > self.config.junction_side_open_threshold:
            exits.append(-45)
        if right > self.config.junction_side_open_threshold:
            exits.append(45)

        front_open = front_distance > self.config.junction_front_open_threshold
        if front_open:
            exits.append(0)

        # Cross-corridor: front AND at least one side open
        if front_open and len(exits) >= 2:
            return True, "cross", exits

        # Open junction: front open, limited side exits
        if front_open:
            return True, "open", exits

        # T-junction: front blocked but at least one side open
        if len(exits) >= 1:
            return True, "T", exits

        return False, "", []

    def update_junction(self, front_distance: float, scan: Dict[int, float], now: float):
        detected, jtype, exits = self.detect_junction(front_distance, scan)
        self.junction_detected = detected
        if detected:
            self.junction_type = jtype
            self.junction_time = now
            self.junction_exit_headings = exits

    def junction_bonus_for_angle(self, angle: int, now: float) -> float:
        """Bonus for heading toward unexplored junction directions."""
        if not self.junction_detected:
            return 0.0
        if now - self.junction_time > 4.0:
            return 0.0
        if angle in self.junction_exit_headings:
            return self.config.junction_unexplored_bonus
        for exit_h in self.junction_exit_headings:
            gap = abs(angle - exit_h)
            if gap <= 45:
                return self.config.junction_unexplored_bonus * (1.0 - gap / 45.0) * self.config.junction_corner_preference
        return 0.0

    def record_near_miss(self, front_distance: float, heading: int):
        """Track close calls for learning."""
        if front_distance <= self.config.near_miss_distance and front_distance > 0:
            self.near_miss_history.append(front_distance)
            self.near_miss_angles.append(heading)

    def get_near_miss_caution_multiplier(self) -> float:
        """Increase caution after a cluster of near-misses."""
        if len(self.near_miss_history) < 3:
            return 1.0
        recent = list(self.near_miss_history)[-3:]
        if all(d < self.config.near_miss_distance * 0.7 for d in recent):
            return self.config.near_miss_caution_multiplier
        return 1.0

    def near_miss_penalty_for_angle(self, angle: int) -> float:
        """Penalize angles recently involved in near-misses."""
        if not self.near_miss_angles:
            return 0.0
        penalty = 0.0
        for miss_angle in self.near_miss_angles:
            if abs(angle - miss_angle) < 30:
                penalty += 1.0
        return -penalty * self.config.near_miss_learned_avoid_gain / max(1, len(self.near_miss_angles))

    def detect_loop(self, front_distance: float, heading: int, now: float) -> bool:
        """Detect when the robot has returned to a previously visited area."""
        self.loop_detection_positions.append((heading, front_distance))
        if len(self.loop_detection_positions) < 15:
            return False
        positions = list(self.loop_detection_positions)
        recent_heading = positions[-1][0]
        recent_front = positions[-1][1]
        matches = 0
        for h, f in positions[:-5]:
            if abs(h - recent_heading) < 20 and abs(f - recent_front) < 10:
                matches += 1
        if matches >= 3 and now - self.last_loop_detection_time > 30.0:
            self.loop_count += 1
            self.last_loop_detection_time = now
            return True
        return False

    def get_loop_escape_bonus(self, angle: int, now: float) -> float:
        """After detecting a loop, strongly bias toward unexplored headings."""
        if self.loop_count == 0 or now - self.last_loop_detection_time > 15.0:
            return 0.0
        recent_headings = [h for h, _ in list(self.loop_detection_positions)[-15:]]
        for rh in recent_headings:
            if abs(angle - rh) < 25:
                return -15.0
        return 20.0

    def predict_openings(self, scan: Dict[int, float], now: float):
        """Predict which directions are likely to open up based on scan trends."""
        if now - self.last_prediction_time < 0.5:
            return
        self.last_prediction_time = now
        for angle in scan:
            hist = self.angle_rate_history.get(angle)
            if not hist or len(hist) < 3:
                continue
            entries = list(hist)
            if len(entries) >= 3:
                recent_trend = entries[-1][1] - entries[0][1]
                self.predicted_openings[angle] = recent_trend
        decay = 0.95
        for angle in list(self.predicted_openings.keys()):
            self.predicted_openings[angle] *= decay
            if abs(self.predicted_openings[angle]) < 0.5:
                del self.predicted_openings[angle]

    def anticipatory_opening_bonus(self, angle: int) -> float:
        """Bonus for heading toward directions predicted to open up."""
        trend = self.predicted_openings.get(angle, 0.0)
        if trend > 5.0:
            return trend * 0.3
        return 0.0

    def update_obstacle_velocity(self, scan: Dict[int, float], now: float):
        """Track how obstacle distances change over time to detect moving obstacles."""
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            if dist <= 0.0:
                continue
            if angle not in self.obstacle_velocity_map:
                self.obstacle_velocity_map[angle] = deque(maxlen=self.config.obstacle_velocity_window)
            self.obstacle_velocity_map[angle].append((now, dist))
        self.moving_obstacles.clear()
        for angle, history in self.obstacle_velocity_map.items():
            if len(history) < 3:
                continue
            entries = list(history)
            dt = entries[-1][0] - entries[0][0]
            if dt <= 0.0:
                continue
            velocity = abs(entries[-1][1] - entries[0][1]) / dt
            if velocity > self.config.obstacle_velocity_threshold:
                self.moving_obstacles.add(angle)

    def is_moving_obstacle(self, angle: int) -> bool:
        """Check if an obstacle at the given angle appears to be moving."""
        return angle in self.moving_obstacles

    def apply_heading_hysteresis(self, new_heading: int, best_score: float, second_score: float) -> int:
        """Prevent jittery heading changes by requiring a meaningful score difference."""
        threshold = self.config.heading_hysteresis_threshold
        if new_heading == self.heading_hysteresis_last:
            self.heading_hysteresis_count = min(self.heading_hysteresis_count + 1, self.config.heading_hysteresis_window)
            return new_heading
        score_gap = best_score - second_score
        if self.heading_hysteresis_count < self.config.heading_hysteresis_window:
            if score_gap < threshold:
                return self.heading_hysteresis_last
        self.heading_hysteresis_last = new_heading
        self.heading_hysteresis_count = 0
        return new_heading

    def wall_following_escape(self, scan: Dict[int, float], now: float) -> Tuple[bool, MotionCommand | None]:
        """Systematic wall-following escape when stuck in complex obstacles."""
        if not self.wall_following_escape_active:
            if now - self.wall_following_escape_start < 5.0:
                return False, None
            self.wall_following_escape_active = True
            self.wall_following_escape_start = now
            left = self.sanitize_distance(scan.get(-45, 0.0))
            right = self.sanitize_distance(scan.get(45, 0.0))
            self.wall_following_escape_direction = -1 if left > right else 1
            return False, None

        elapsed = now - self.wall_following_escape_start
        if elapsed > 8.0:
            self.wall_following_escape_active = False
            self.wall_following_escape_start = -math.inf
            return False, None

        direction = self.wall_following_escape_direction
        speed = self.config.escape_turn_speed * 0.6
        if direction < 0:
            left_s = -speed * 0.3
            right_s = speed
        else:
            left_s = speed
            right_s = -speed * 0.3

        heading = direction * 60
        return True, MotionCommand(
            mode="escape",
            left_speed=left_s,
            right_speed=right_s,
            heading=heading,
            colour=self.motion_colour_for("escape", heading, 0.0, 0.0),
        )

    def plan_escape_arc(self, scan: Dict[int, float], heading: int, now: float) -> MotionCommand:
        """Execute a curved escape trajectory instead of straight reverse."""
        direction = -1 if heading < 0 else 1
        arc = self.config.escape_arc_radius
        base_reverse = self.config.escape_reverse_speed
        reverse_speed = base_reverse * (1.0 - arc * 0.3)
        turn_component = base_reverse * arc * direction
        left_speed = -(reverse_speed + turn_component * 0.5)
        right_speed = -(reverse_speed - turn_component * 0.5)
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))
        return MotionCommand(
            mode="escape",
            left_speed=left_speed,
            right_speed=right_speed,
            heading=heading,
            colour=self.motion_colour_for("escape", heading, 0.0, 0.0),
        )

    def record_environmental_memory(self, front_distance: float, heading: int, now: float, mode: str):
        """Remember notable positions and events for narrative storytelling."""
        tag = ""
        if mode == "escape":
            tag = "escape"
        elif mode == "dead_end_recovery":
            tag = "dead_end"
        elif front_distance > self.config.open_space_distance:
            tag = "open"
        elif front_distance < self.config.danger_distance:
            tag = "danger"
        elif mode == "brave_push":
            tag = "push"
        if tag:
            self.environmental_memory.append((front_distance, heading, tag))

    def generate_narrative_quip(self, now: float) -> str:
        """Generate running commentary about the robot's journey."""
        if now - self.last_narrative_time < self.config.narrative_cooldown_s:
            return ""
        if len(self.environmental_memory) < 3:
            return ""
        recent = list(self.environmental_memory)[-self.config.environmental_memory_size:]
        escapes = sum(1 for _, _, t in recent if t == "escape")
        dead_ends = sum(1 for _, _, t in recent if t == "dead_end")
        open_spaces = sum(1 for _, _, t in recent if t == "open")
        dangers = sum(1 for _, _, t in recent if t == "danger")
        total = len(recent)
        personality = self.get_personality_label()
        candidates = []
        if escapes > dead_ends and escapes >= 3:
            candidates.append(f"recent history: {escapes} escapes, {dead_ends} dead ends — {personality} is an escape artist")
            candidates.append(f"I've escaped {escapes} times in the last {self.config.environmental_story_window:.0f}s. the walls keep losing.")
        if dead_ends >= 3:
            candidates.append(f"{dead_ends} dead ends recently. {personality} is starting to think this room hates me")
            candidates.append(f"dead end count: {dead_ends}. at this point I'm mapping the walls, not the exits")
        if open_spaces >= 4 and dangers <= 1:
            candidates.append(f"been in open space {open_spaces} times. {personality} is enjoying the freedom")
            candidates.append(f"mostly open spaces lately ({open_spaces}/{total}). this must be the good part of the room")
        if dangers >= 3:
            candidates.append(f"{dangers} close calls recently. {personality} needs a nap")
            candidates.append(f"danger zone: {dangers} encounters and counting. {personality} living on the edge")
        escape_ratio = escapes / max(1, total)
        if escape_ratio > 0.4 and total >= 5:
            candidates.append(f"{int(escape_ratio*100)}% of my recent life has been escaping. {personality} is rethinking career choices")
        if self.total_distance_traveled > 2000 and len(self.recent_narrative_topics) > 0:
            candidates.append(f"traveled {int(self.total_distance_traveled/100)}m so far. still no idea where I'm going. {personality} and proud")
        if self.loop_count > 0:
            candidates.append(f"loop counter: {self.loop_count}. I've been here before, haven't I? {personality} is confused")
        if not candidates:
            return ""
        quip = candidates[int(abs(now * 3571 + self.total_loops * 13)) % len(candidates)]
        for topic in self.recent_narrative_topics:
            if topic in quip:
                return ""
        self.last_narrative_time = now
        self.recent_narrative_topics.append(quip[:40])
        return quip

    def detect_side_corridor(self, front_distance: float, scan: Dict[int, float], now: float):
        """Detect when one side opens up as an escape route (side corridor)."""
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            self.side_corridor_detected = False
            return
        asymmetry = abs(left - right)
        if asymmetry < self.config.side_corridor_asymmetry_min:
            self.side_corridor_detected = False
            return
        open_side = -45 if left > right else 45
        open_dist = max(left, right)
        if open_dist > self.config.side_corridor_open_threshold:
            self.side_corridor_detected = True
            self.side_corridor_heading = open_side
            self.side_corridor_time = now
        elif now - self.side_corridor_time > self.config.side_corridor_memory_s:
            self.side_corridor_detected = False
            self.side_corridor_heading = 0

    def side_corridor_bonus_for_angle(self, angle: int, now: float) -> float:
        """Bonus for heading toward a detected side corridor."""
        if not self.side_corridor_detected:
            return 0.0
        if now - self.side_corridor_time > self.config.side_corridor_memory_s:
            return 0.0
        if angle == self.side_corridor_heading:
            return self.config.side_corridor_bonus
        angle_gap = abs(angle - self.side_corridor_heading)
        if angle_gap <= 45:
            return self.config.side_corridor_bonus * (1.0 - angle_gap / 45.0)
        return 0.0

    def adapt_exploration_strategy(self, front_distance: float, scan: Dict[int, float], now: float):
        """Dynamically switch between exploration strategies based on environment."""
        self.strategy_eval_count += 1
        if self.strategy_eval_count < self.config.strategy_adapt_window:
            return

        self.strategy_eval_count = 0
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))

        edge_reward = self.strategy_reward.get("edge_follow", 0.0)
        center_reward = self.strategy_reward.get("center", 0.0)

        in_corridor = (
            left > 0 and right > 0
            and left < self.config.wall_following_threshold
            and right < self.config.wall_following_threshold
            and front_distance > self.config.caution_distance
        )

        if in_corridor and self.current_exploration_strategy != "edge_follow":
            if edge_reward >= center_reward:
                self.current_exploration_strategy = "edge_follow"
                self.last_strategy_change_time = now
        elif front_distance > self.config.open_space_distance and self.current_exploration_strategy != "center":
            if center_reward >= edge_reward:
                self.current_exploration_strategy = "center"
                self.last_strategy_change_time = now
        elif self.current_exploration_strategy != "balanced":
            self.current_exploration_strategy = "balanced"
            self.last_strategy_change_time = now
        elif front_distance > self.config.open_space_distance and self.current_exploration_strategy != "center":
            if center_reward >= edge_reward:
                self.current_exploration_strategy = "center"
                self.last_strategy_change_time = now
        elif self.current_exploration_strategy != "balanced":
            self.current_exploration_strategy = "balanced"
            self.last_strategy_change_time = now

        self.strategy_reward["edge_follow"] = 0.0
        self.strategy_reward["center"] = 0.0

    def apply_exploration_strategy_steer(self, steer: float, front_distance: float, scan: Dict[int, float], now: float) -> float:
        """Apply strategy-specific steering adjustments."""
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        strategy = self.current_exploration_strategy

        if strategy == "edge_follow":
            pref = self.config.edge_follow_side_preference
            if pref < 0 and left > 0 and left < self.config.wall_following_threshold * 1.5:
                target_dist = self.config.wall_following_threshold * 0.6
                error = left - target_dist
                steer -= error * self.config.edge_follow_gain
                self.strategy_reward["edge_follow"] = self.strategy_reward.get("edge_follow", 0.0) + self.config.strategy_edge_reward
            elif pref > 0 and right > 0 and right < self.config.wall_following_threshold * 1.5:
                target_dist = self.config.wall_following_threshold * 0.6
                error = right - target_dist
                steer += error * self.config.edge_follow_gain
                self.strategy_reward["edge_follow"] = self.strategy_reward.get("edge_follow", 0.0) + self.config.strategy_edge_reward

        elif strategy == "center":
            if left > 0 and right > 0:
                total = left + right
                if total > 0:
                    imbalance = (left - right) / total
                    steer -= imbalance * self.config.center_lane_gain
                    self.strategy_reward["center"] = self.strategy_reward.get("center", 0.0) + self.config.strategy_center_reward

        if now - self.last_random_exploration_time > self.config.strategy_random_turn_interval:
            if self.current_exploration_strategy == "random" or (self.current_exploration_strategy == "balanced" and self.total_loops % 300 == 0):
                import random
                random_angle = random.choice([-60, -45, -30, 30, 45, 60])
                max_angle = max(abs(a) for a in self.config.scan_angles) or 1
                steer = max(-1.0, min(1.0, random_angle / max_angle))
                self.last_random_exploration_time = now

        return steer

    def should_escalate_recovery(self, now: float) -> bool:
        """Check if we should move to the next recovery stage."""
        return now - self.recovery_stage_time >= self.config.recovery_stage_duration

    def advance_recovery_stage(self, now: float):
        """Move to a more aggressive recovery maneuver."""
        self.recovery_stage = min(3, self.recovery_stage + 1)
        self.recovery_stage_time = now

    def reset_recovery(self):
        """Reset recovery state when navigation is successful."""
        self.recovery_stage = 0
        self.recovery_stage_time = -math.inf

    def record_notable_event(self, event: str, now: float):
        """Remember notable events for narrative continuity in humour."""
        self.last_notable_event = event
        self.last_notable_event_time = now
        self.event_memory.append(f"{event}@{now:.0f}")

    def compute_option_value(self, angle: int, scan: Dict[int, float]) -> float:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        angle_factor = abs(angle) / max_angle
        forward_dist = self.sanitize_distance(scan.get(angle, 0.0))
        if forward_dist <= 0.0:
            return 0.0
        options = 0.0
        range_deg = self.config.option_value_angle_range
        for a, d in scan.items():
            d = self.sanitize_distance(d)
            if d > 0.0 and abs(a - angle) <= range_deg:
                options += d / self.config.cruise_distance
        forward_value = forward_dist / self.config.cruise_distance
        return (forward_value * (1.0 - self.config.option_value_weight) +
                options * self.config.option_value_weight) * (1.0 - angle_factor * 0.3)

    def detect_side_closing(self, scan: Dict[int, float]) -> float:
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return 0.0
        total_side = left + right
        self.side_closing_history.append(total_side)
        if len(self.side_closing_history) < 2:
            return 0.0
        history = list(self.side_closing_history)
        closing = 0
        for i in range(1, len(history)):
            if history[i] < history[i - 1] - self.config.side_closing_threshold * 0.3:
                closing += 1
        return min(1.0, closing / max(1, len(history) - 1))

    def side_asymmetry_bias(self, angle: int, ordered_scan: Tuple[Tuple[int, float], ...], front_distance: float) -> float:
        """Reward headings that point toward the more open side of an asymmetric environment.
        When one side is significantly more open than the other, bias toward that direction."""
        scan_dict = {a: d for a, d in ordered_scan}
        left = self.sanitize_distance(scan_dict.get(-45, 0.0))
        right = self.sanitize_distance(scan_dict.get(45, 0.0))
        if left <= 0.0 or right <= 0.0:
            return 0.0
        asymmetry = right - left
        if abs(asymmetry) < 15.0:
            return 0.0
        normalized = asymmetry / max(left + right, 1.0)
        if angle > 0 and asymmetry > 0:
            return abs(normalized) * 12.0
        if angle < 0 and asymmetry < 0:
            return abs(normalized) * 12.0
        return -abs(normalized) * 6.0

    def compute_novelty_bonus(self, angle: int) -> float:
        if len(self.flow_memory) < self.config.novelty_min_flow_entries:
            return 0.0
        radius = self.config.novelty_angular_radius
        for h, _ in self.flow_memory:
            if abs(h - angle) < radius:
                return 0.0
        return self.config.novelty_bonus

    def get_quip(self, mode: str, front_distance: float, heading: int, now: float) -> str:
        if now - self.last_quip_time < self.config.quip_cooldown_s:
            return ""
        self.last_quip_time = now
        personality = self.get_personality_label()
        streak = self.success_streak
        stress = self.stress
        candidates = []
        closing = self.last_side_closing > 0.5
        terrain = self.terrain_class
        in_corridor = (
            self.last_scan.get(-45, 0.0) > 0.0
            and self.last_scan.get(45, 0.0) > 0.0
            and self.last_scan.get(-45, 0.0) < self.config.wall_following_threshold
            and self.last_scan.get(45, 0.0) < self.config.wall_following_threshold
            and front_distance > self.config.caution_distance
        )
        shape = self.obstacle_shape
        if mode == "escape":
            self.escape_count += 1
            esc = self.escape_count
            if esc == 1:
                candidates = [
                    "eek! first bump — " + personality + " but not defeated",
                    "tactical retreat! " + personality + " mode engaged",
                    "oops, that wall came out of nowhere",
                    "bonk! " + personality + " learning the room layout the hard way",
                ]
            elif esc < 5:
                candidates = [
                    "escape #" + str(esc) + "! " + personality + " and getting crafty",
                    "not again... " + personality + " but persistent",
                    "wall " + str(esc) + ", me 0. adjusting " + personality + " strategy",
                    "evasive #" + str(esc) + "! this room is a maze",
                    "reverse! reverse! " + personality + " backing away from that",
                ]
            elif esc < 10:
                candidates = [
                    "ESCAPE #" + str(esc) + "! " + personality + " and slightly annoyed",
                    "wall #" + str(esc) + " wins this round. " + personality + " rage building",
                    "are these walls multiplying? escape #" + str(esc) + ", " + personality,
                    "at this rate I'll have memorized every wall. #" + str(esc) + ", " + personality,
                ]
            else:
                candidates = [
                    "ESCAPE #" + str(esc) + "!! " + personality + " and questioning reality",
                    "wall #" + str(esc) + "... I'm starting to think this is personal",
                    "at this point I'm just decorating with escape marks. #" + str(esc) + ", " + personality,
                    "I've bumped into more walls than a Roomba at a furniture store. #" + str(esc),
                ]
            if stress > 0.5:
                bar = "\u2588\u2588\u2588\u2588" if stress > 0.8 else "\u2588\u2588\u2591\u2591" if stress > 0.6 else "\u2588\u2591\u2591\u2591"
                candidates.append("stress level: " + bar + " \u2014 " + personality + " and sweating")
            if in_corridor:
                candidates.append("corridor escape! squeezing out of this tight spot #" + str(esc))
            if self.recovery_stage >= 2:
                candidates.append("recovery stage " + str(self.recovery_stage) + "! " + personality + " getting desperate")
                candidates.append("escalating to stage " + str(self.recovery_stage) + ". this is fine.")
            if self.total_distance_traveled > 3000:
                candidates.append(str(esc) + " escapes after " + str(int(self.total_distance_traveled / 100)) + "m. " + personality + " not bad for a robot")
            self.record_notable_event("escape_" + str(esc), now)
        elif mode == "peek_pounce":
            self.pounce_count += 1
            side = "L" if heading < 0 else "R"
            candidates = [
                "predatory lane-change! " + side + " looks tasty (" + personality + ")",
                "pounce #" + str(self.pounce_count) + "! " + side + " side is calling",
                "ooh, " + side + " side has room \u2014 " + personality + " going for it",
                "gap detected! " + side + " looks promising, " + personality + " commit",
                "sneaky " + side + " shift! " + personality + " slipping through",
            ]
            if streak >= 3:
                candidates.append("on a streak of " + str(streak) + "! " + side + " side won't know what hit it")
            if personality == "bold":
                candidates.append("YOLO into the " + side + " gap! " + personality + " style")
        elif mode == "brave_push":
            self.push_count += 1
            push = self.push_count
            candidates = [
                "CHARGE #" + str(push) + "! (what's the worst that could happen?)",
                "brave push #" + str(push) + "! obstacle, meet face \u2014 " + personality + " mode",
                "FULL SEND #" + str(push) + "! hopefully it moves... (" + personality + ")",
                "ramming speed #" + str(push) + "! beep beep \u2014 " + personality,
            ]
            if push > 3:
                candidates.append("push #" + str(push) + "... at this point it's a personal vendetta")
            if personality == "bold":
                candidates.append("BOLD CHARGE #" + str(push) + "! fear is just a suggestion")
            elif personality == "cautious":
                candidates.append("gentle push #" + str(push) + "... please move? " + personality + " asking nicely")
        elif mode == "dead_end_recovery":
            self.dead_end_count += 1
            side = "L" if heading < 0 else "R"
            candidates = [
                "dead end #" + str(self.dead_end_count) + "... " + personality + " and turning " + side,
                "this wall again? #" + str(self.dead_end_count) + ". " + personality + " sigh",
                "cul-de-sac #" + str(self.dead_end_count) + ". time for a " + side + " pivot, " + personality,
                "plot twist: another dead end. #" + str(self.dead_end_count) + ", " + personality,
            ]
            if self.dead_end_count > 3:
                candidates.append("dead end #" + str(self.dead_end_count) + "... I'm mapping this place as 'the bad room'")
            if self.dead_end_count > 6:
                candidates.append("dead end #" + str(self.dead_end_count) + "... I've accepted my fate in this maze")
            if self.dead_end_count > 10:
                candidates.append("dead end #" + str(self.dead_end_count) + ". at this point I'm the maze")
                candidates.append("dead end #" + str(self.dead_end_count) + ". I wonder if the walls enjoy this too")
            if personality == "panicked":
                candidates.append("AAAAH dead end #" + str(self.dead_end_count) + "! " + personality + " spin spin spin")
            if self.escape_count > self.dead_end_count * 2:
                candidates.append("dead end #" + str(self.dead_end_count) + " but at least I'm good at escaping (" + str(self.escape_count) + " escapes)")
            self.record_notable_event("dead_end_" + str(self.dead_end_count), now)
        elif mode == "drive":
            if in_corridor:
                candidates = [
                    "corridor cruising \u2014 " + personality + " parallel parking champion",
                    "wall-hugging like a pro, " + personality + " style",
                    "both sides tight but I'm threading through, " + personality,
                    "corridor mode: " + personality + " staying centered",
                ]
                if streak >= 3:
                    candidates.append("corridor streak of " + str(streak) + "! " + personality + " corridor master")
            elif front_distance > self.config.open_space_distance:
                candidates = [
                    "open road! " + personality + " cruising at " + str(int(front_distance)) + "cm",
                    "clear skies ahead~ " + personality + " and feeling good",
                    "nothing but open space for " + str(int(front_distance)) + "cm. " + personality + " zen mode",
                    "freedom! " + str(int(front_distance)) + "cm of pure possibility, " + personality,
                ]
                if streak >= 5:
                    candidates.append("streak=" + str(streak) + "! " + personality + " owning this open space")
            elif front_distance > self.config.cruise_distance:
                candidates = [
                    "coasting through at " + str(int(front_distance)) + "cm, " + personality,
                    "comfortable cruise, " + personality + " vibes",
                    "smooth sailing, " + str(int(front_distance)) + "cm clear, " + personality,
                ]
            elif front_distance > self.config.caution_distance:
                if streak >= 5:
                    candidates.append("streak=" + str(streak) + "! navigating " + str(int(front_distance)) + "cm like a " + personality + " pro")
                elif closing:
                    candidates = [
                        "sides closing in! " + str(int(front_distance)) + "cm, " + personality + " staying alert",
                        "gap narrowing at " + str(int(front_distance)) + "cm, " + personality + " and cautious",
                    ]
                else:
                    candidates = [
                        "threading the needle at " + str(int(front_distance)) + "cm, " + personality,
                        "careful but confident \u2014 " + str(int(front_distance)) + "cm, " + personality,
                    ]
            else:
                if personality == "bold":
                    candidates = [
                        "BOLD but cautious: " + str(int(front_distance)) + "cm and not flinching",
                        "close but " + personality + " \u2014 " + str(int(front_distance)) + "cm, no sweat",
                    ]
                elif personality == "panicked":
                    candidates = [
                        "TOO CLOSE! " + str(int(front_distance)) + "cm! " + personality,
                        "personal space violation: " + str(int(front_distance)) + "cm, " + personality,
                    ]
                else:
                    candidates = [
                        "caution zone: " + str(int(front_distance)) + "cm, " + personality + " and alert",
                        "close quarters! " + str(int(front_distance)) + "cm, " + personality + " mode",
                        "tight squeeze at " + str(int(front_distance)) + "cm, " + personality + " inching forward",
                    ]
            if terrain == "cluttered":
                candidates.append("cluttered chaos! " + personality + " dodging at " + str(int(front_distance)) + "cm")
            elif terrain == "open" and front_distance > self.config.cruise_distance:
                candidates.append("open terrain detected! " + personality + " speed demon mode")
            if shape == "wall" and front_distance > self.config.caution_distance:
                candidates.append("wall ahead but I see a way around, " + personality + " style")
            elif shape == "corner" and front_distance > self.config.caution_distance:
                candidates.append("corner detected! " + personality + " taking the open side")
            elif shape == "isolated" and front_distance > self.config.caution_distance:
                candidates.append("isolated obstacle — easy dodge for " + personality)
            if self.dead_end_predicted:
                candidates.append("dead-end forming ahead... " + personality + " plotting escape route")
            if streak >= 10 and front_distance > self.config.cruise_distance:
                candidates.append("UNSTOPPABLE! streak=" + str(streak) + " " + personality + " is a navigation god")
                candidates.append("they said it couldn't be done. streak=" + str(streak) + " " + personality + " proved them wrong")
            if self.total_distance_traveled > 5000 and self.total_loops % 200 == 0:
                candidates.append("odometer: " + str(int(self.total_distance_traveled)) + "cm and counting")
                candidates.append("that's " + str(int(self.total_distance_traveled / 100)) + "m of pure exploration")
            if self.last_notable_event and now - self.last_notable_event_time < 15.0:
                event = self.last_notable_event
                if event == "wiggling":
                    candidates.append("still wiggling... " + personality + " and not giving up")
                    candidates.append("the wiggle strategy isn't working but I'm committed now")
                elif event == "reverse_spin":
                    candidates.append("reverse spin didn't help but at least it was dramatic")
                    candidates.append("spun in reverse. walls still winning. " + personality)
                elif event == "full_spin":
                    candidates.append("full 360 spin! " + personality + " dizzy but determined")
                    candidates.append("spun all the way around. same room, different angle")
        if self.escape_count > 0 and self.escape_count % 10 == 0 and mode == "drive" and front_distance > self.config.caution_distance:
            candidates.append("milestone: " + str(self.escape_count) + " escapes and still going. " + personality + " resilient")
            candidates.append("fun fact: I've escaped " + str(self.escape_count) + " times. wall count: unknown. " + personality)
        if self.junction_detected and now - self.junction_time < 3.0 and mode == "drive":
            jtype = self.junction_type
            exits = len(self.junction_exit_headings)
            if jtype == "T":
                candidates.append("T-junction! " + str(exits) + " way" + ("s" if exits > 1 else "") + " open — " + personality + " choosing wisely")
                candidates.append("T-junction ahead. " + personality + " taking the scenic route")
            elif jtype == "cross":
                candidates.append("cross-corridor! " + str(exits) + " exits — " + personality + " at a crossroads")
                candidates.append("intersection! " + personality + " feeling like a real explorer")
            elif jtype == "open":
                candidates.append("open junction! " + str(exits) + " paths — " + personality + " spoiled for choice")
        if self.side_corridor_detected and now - self.side_corridor_time < 3.0 and mode == "drive":
            side = "L" if self.side_corridor_heading < 0 else "R"
            candidates.append("side corridor open on " + side + "! " + personality + " sees an escape route")
            candidates.append(side + " side looks promising — " + personality + " checking it out")
            if personality == "bold":
                candidates.append(side + " side is wide open! " + personality + " going for it")
            elif personality == "cautious":
                candidates.append(side + " side might be a way out... " + personality + " proceeding carefully")
        if self.tactical_patience_active and mode == "drive":
            candidates.append("tactical pause... " + personality + " assessing the situation")
            candidates.append("wait, let me think about this... " + personality + " mode")
        if len(self.near_miss_history) >= 3 and mode == "drive":
            recent = list(self.near_miss_history)[-3:]
            if all(d < self.config.near_miss_distance * 0.7 for d in recent):
                candidates.append("that was too close! " + personality + " dialing up caution")
                candidates.append("near-miss cluster detected. " + personality + " playing it safe for a bit")
        strategy_tag = self.current_exploration_strategy
        if strategy_tag == "edge_follow" and mode == "drive":
            candidates.append("wall-hugging mode — " + personality + " staying close to the action")
            candidates.append("edge-follow strategy. " + personality + " like a Roomba with ambition")
        elif strategy_tag == "center" and mode == "drive":
            candidates.append("center-lane cruising — " + personality + " owning the middle")
            candidates.append("staying centered. " + personality + " no wall left/right bias today")
        if self.predicted_openings and mode == "drive":
            best_angle = max(self.predicted_openings, key=self.predicted_openings.get)
            if self.predicted_openings[best_angle] > 10.0:
                candidates.append("sensing opening at " + str(best_angle) + "° — " + personality + " following the flow")
                candidates.append("things are getting roomier that way... " + personality + " trusting the trend")
        if self.patience_reassess_count > 3 and mode == "drive" and front_distance > self.config.caution_distance:
            candidates.append("patience reassessment #" + str(self.patience_reassess_count) + " — " + personality + " getting better at this")
        if self.loop_count > 0 and mode == "drive" and now - self.last_loop_detection_time < 15.0:
            loop_quips = [
                "wait, I've been here before... loop #" + str(self.loop_count) + ", " + personality + " is confused",
                "déjà vu! loop #" + str(self.loop_count) + " — " + personality + " needs a breadcrumb trail",
                "loop #" + str(self.loop_count) + "! I'm basically a Roomba with delusions of grandeur",
                "this room is a circle and I am the circumference. loop #" + str(self.loop_count) + ", " + personality,
            ]
            candidates.extend(loop_quips)
            if self.loop_count >= 3:
                candidates.append("loop #" + str(self.loop_count) + "... I should start leaving markers. " + personality + " getting desperate")
            if self.loop_count >= 5:
                candidates.append("loop #" + str(self.loop_count) + ". at this point I'm not exploring, I'm pacing")
        if self.wall_following_escape_active and mode == "escape":
            candidates.append("wall-following escape! " + personality + " hugging the right wall like a pro")
            candidates.append("systematic wall-follow mode. " + personality + " finally using strategy")
        if self.moving_obstacles and mode == "drive":
            moving_list = ", ".join(str(a) + "°" for a in sorted(self.moving_obstacles))
            candidates.append("something's moving at " + moving_list + "! " + personality + " staying alert")
            candidates.append("moving obstacle detected. " + personality + " is not alone in here")

        # Gen22: boustrophedon quips
        if self.boustrophedon_active and mode == "drive":
            lane = self.boustrophedon_current_lane
            candidates.append("lawnmower mode! lane #" + str(lane) + " — " + personality + " covering every inch")
            candidates.append("systematic coverage in progress. " + personality + " leaving no stone unturned")
            if lane > 3:
                candidates.append("lane #" + str(lane) + " of the great grid sweep. " + personality + " methodical")

        # Gen22: straight-line momentum quips
        if self.straight_line_momentum > 0.6 and mode == "drive" and front_distance > self.config.cruise_distance:
            candidates.append("cruising straight with momentum! " + personality + " locked in")
            candidates.append("straight-line flow state. " + personality + " nothing can stop me now")
            if self.straight_line_momentum > 0.8:
                candidates.append("MAXIMUM straight-line momentum! " + personality + " FREIGHT TRAIN MODE")

        # Gen22: room awareness quips
        if self.room_boundary_confirmed and mode == "drive" and self.total_loops % 200 == 0:
            openness = self.get_room_openness()
            candidates.append("room mapped at " + str(int(openness * 100)) + "% openness. " + personality + " knows this place")

        # Gen22: playfulness quips
        if self.playfulness_level > 0.7 and mode == "drive" and front_distance > self.config.open_space_distance:
            candidates.append("feeling playful today! " + personality + " let's have some fun")
            candidates.append("energy level high! " + personality + " ready to zoom")

        # Gen23: topological exploration quips
        if self.topological_nodes and mode == "drive" and self.total_loops % 150 == 0:
            node_count = len(self.topological_nodes)
            unvisited = self.count_unvisited_nodes()
            candidates.append("mapped " + str(node_count) + " areas, " + str(unvisited) + " still unexplored — " + personality + " curious")
            if unvisited > 0:
                candidates.append(str(unvisited) + " uncharted territories remain. " + personality + " on the case")
        if self.last_area_classification != "unknown" and mode == "drive" and self.total_loops % 200 == 0:
            candidates.append("currently in a " + self.last_area_classification + ". " + personality + " adapting")
        if self.vfh_valleys and mode == "drive" and front_distance > self.config.cruise_distance:
            best_valley = max(self.vfh_valleys, key=lambda v: v[3])
            candidates.append("VFH sees a " + str(best_valley[3]) + "-sector valley at " + str(int(best_valley[2])) + "° — " + personality + " following the flow")

        narrative = self.generate_narrative_quip(now)
        if narrative and mode == "drive":
            candidates.append(narrative)
        if now - self.last_commentary_time > self.config.running_commentary_interval and mode == "drive" and self.total_loops > 50:
            if self.total_distance_traveled > 500:
                dist_m = int(self.total_distance_traveled / 100)
                candidates.append("status report: " + str(dist_m) + "m explored, " + str(self.escape_count) + " escapes, " + personality + " still curious")
            if self.terrain_class != "unknown":
                candidates.append("current terrain assessment: " + self.terrain_class + ". " + personality + " accepts this")
            self.last_commentary_time = now
        if self.total_loops > 0 and self.total_loops % 500 == 0 and mode == "drive":
            candidates.append("loop " + str(self.total_loops) + "! " + personality + " veteran explorer")
            candidates.append(str(self.total_loops) + " decision cycles and I still don't know where I am. " + personality)
        if self.escape_count > 0 and self.escape_count % 5 == 0 and mode == "drive" and self.last_achievement != "escape_" + str(self.escape_count):
            self.last_achievement = "escape_" + str(self.escape_count)
            self.achievement_count += 1
            self.last_achievement_time = now
            if self.escape_count == 5:
                candidates.append("achievement unlocked: First Steps (5 escapes)")
            elif self.escape_count == 10:
                candidates.append("achievement unlocked: Escape Artist (10 escapes)")
            elif self.escape_count == 25:
                candidates.append("achievement unlocked: Houdini would be proud (25 escapes)")
            elif self.escape_count == 50:
                candidates.append("achievement unlocked: Professional Escapee (50 escapes)")
            elif self.escape_count == 100:
                candidates.append("achievement unlocked: The Walls Fear Me (100 escapes)")
        if self.success_streak >= 15 and self.last_achievement != "streak_15":
            self.last_achievement = "streak_15"
            self.achievement_count += 1
            self.last_achievement_time = now
            candidates.append("achievement unlocked: Unstoppable Force (15-streak)")
        if self.total_distance_traveled > 10000 and self.last_achievement != "distance_100m":
            self.last_achievement = "distance_100m"
            self.achievement_count += 1
            self.last_achievement_time = now
            candidates.append("achievement unlocked: Marathon Runner (100m traveled)")
        if self.navigation_insights > 0 and self.last_achievement != "insight_" + str(self.navigation_insights):
            self.last_achievement = "insight_" + str(self.navigation_insights)
            self.achievement_count += 1
            self.last_achievement_time = now
            candidates.append("achievement unlocked: Growing Smarter (" + str(self.navigation_insights) + " insights)")
        if now - self.last_self_aware_quip_time > self.config.self_aware_quip_cooldown and self.total_loops > 100:
            self_aware_candidates = []
            if self.escape_count > 3 and self.escape_count < 20:
                self_aware_candidates.append("fun fact: I've learned that " + str(self.escape_count) + " walls don't mean I'm lost, just... exploring creatively")
            if len(self.success_map) > 2:
                self_aware_candidates.append("I'm actually building a mental map — " + str(len(self.success_map)) + " directions catalogued so far")
            if self.terrain_class in ("open", "cluttered"):
                self_aware_candidates.append("I've decided this room is " + self.terrain_class + ". I don't know why that makes me feel things")
            if self.dead_end_count > 5:
                self_aware_candidates.append("I've hit " + str(self.dead_end_count) + " dead ends. at this point I'm not lost, I'm doing field research")
            if self.personality_boldness > 0.7 and self.success_streak > 5:
                self_aware_candidates.append("I'm feeling unusually bold today. it's either the success streak or a software bug")
            if self.stress < 0.1 and self.success_streak > 3:
                self_aware_candidates.append("stress level: zen. I think I finally understand this room. or I'm in a different room. either way, vibes")
            if self_aware_candidates:
                quip = self_aware_candidates[int(abs(now * 5381 + self.achievement_count * 31)) % len(self_aware_candidates)]
                for _ in range(min(2, len(self_aware_candidates))):
                    if quip not in self.recent_quips and quip not in self.self_aware_quip_history:
                        break
                    quip = self_aware_candidates[int(abs(now * 4201 + self.achievement_count * 17)) % len(self_aware_candidates)]
                self.self_aware_quip_history.append(quip)
                self.last_self_aware_quip_time = now
                return quip
        if self.obstacle_cluster_detected and mode == "drive":
            tightness = self.obstacle_cluster_tightness
            if tightness > 0.7:
                candidates.append("tight cluster ahead! " + personality + " threading the needle")
                candidates.append("obstacle city! " + personality + " navigating the maze within the maze")
            elif tightness > 0.4:
                candidates.append("clustered obstacles — " + personality + " picking through carefully")
        if self.flow_state == "flow" and mode == "drive":
            candidates.append("in the zone! " + personality + " riding the flow")
            candidates.append("flow state activated — " + personality + " is pure navigation")
            if self.success_streak >= 3:
                candidates.append("streak=" + str(self.success_streak) + " and flowing. " + personality + " is unstoppable")
        elif self.flow_state == "struggle" and mode == "drive":
            candidates.append("struggling today... " + personality + " needs a break")
            candidates.append("nothing's going right. " + personality + " questioning everything")
            if self.escape_count > 2:
                candidates.append(str(self.escape_count) + " escapes and counting. " + personality + " is not having a good time")
        if closing and mode == "drive":
            candidates = [q + " [closing]" for q in candidates] if candidates else ["sides closing in!"]
        if not candidates:
            return ""
        quip = candidates[int(abs(now * 7919 + heading * 31)) % len(candidates)]
        for _ in range(min(3, len(candidates))):
            if quip not in self.recent_quips:
                break
            quip = candidates[int(abs(now * 6271 + heading * 17)) % len(candidates)]
        self.recent_quips.append(quip)
        return quip

    def select_escape_angle(self, scan: Dict[int, float]) -> int:
        best_angle = 0
        best_score = -math.inf
        for angle in self.config.multi_angle_escape_angles:
            dist = self.sanitize_distance(scan.get(angle, 0.0))
            if dist <= 0.0:
                continue
            score = dist + self.config.escape_angle_preference_bonus * (abs(angle) / 60.0)
            if score > best_score:
                best_score = score
                best_angle = angle
        if best_angle == 0:
            best_angle = -80 if sum(self.recent_turns) >= 0 else 80
        return best_angle

    def apply_corner_negotiation(self, heading: int, front_distance: float, speed: float) -> Tuple[int, float]:
        if not self.corner_detected:
            return heading, speed
        open_side = self.corner_side
        heading = int(open_side * self.config.corner_turn_ahead_gain + heading * (1.0 - self.config.corner_turn_ahead_gain))
        speed *= self.config.corner_slowdown_factor
        return heading, speed

    def apply_gap_width_filter(self, angle: int, scan: Dict[int, float], score: float) -> float:
        width = self.estimate_gap_width(angle, scan)
        w = self.get_scoring_weight("gap_width")
        if width < self.config.min_gap_width:
            return score - self.config.gap_reject_penalty * w
        return score + self.config.gap_commitment_bonus * (width / self.config.min_gap_width) * w

    def apply_side_obstacle_avoidance(self, steer: float, front_distance: float, scan: Dict[int, float]) -> float:
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        check_distance = self.config.side_obstacle_proximity_distance if front_distance > self.config.preemptive_side_avoid_distance else self.config.preemptive_side_avoid_distance
        if left > 0.0 and left < check_distance:
            steer += self.config.side_obstacle_avoidance_gain * (1.0 - left / check_distance)
        if right > 0.0 and right < check_distance:
            steer -= self.config.side_obstacle_avoidance_gain * (1.0 - right / check_distance)
        return max(-1.0, min(1.0, steer))

    def _init_scoring_weights(self):
        """Initialize adaptive scoring term weights to 1.0 (baseline)."""
        for term in ("corridor_support", "distance_advantage", "frontier",
                     "danger_heatmap", "spatial_memory", "flow_memory",
                     "danger_zone", "option_value", "novelty", "escape_outcome",
                     "side_asymmetry", "persistence", "escape_direction",
                     "corridor_exit", "success_map", "terrain_memory",
                     "junction", "near_miss", "loop_escape", "anticipatory",
                     "side_corridor", "moving_obstacle", "gap_width",
                     "coverage", "frontier_gen18", "straight_line",
                     "path_smoothness", "grid_novelty", "grid_frontier",
                      "landmark", "straight_lock", "frontier_directed",
                      "dead_zone", "straight_momentum",
                      "topological", "vfh", "exploration_return", "exploration_direction"):
            self.scoring_term_weights[term] = 1.0

    def record_scoring_reward(self, term: str, reward: float):
        """Record whether a scoring term contributed to a good or bad decision."""
        if term not in self.scoring_term_rewards:
            self.scoring_term_rewards[term] = deque(maxlen=self.config.scoring_reward_window)
        self.scoring_term_rewards[term].append(reward)

    def adapt_scoring_weights(self):
        """Adjust scoring term weights based on recent rewards.
        Terms that correlate with good outcomes get boosted; bad ones get reduced."""
        lr = self.config.scoring_term_learning_rate
        floor = self.config.scoring_weight_floor
        ceiling = self.config.scoring_weight_ceiling
        for term, rewards in self.scoring_term_rewards.items():
            if len(rewards) < 4:
                continue
            avg_reward = sum(rewards) / len(rewards)
            current = self.scoring_term_weights.get(term, 1.0)
            target = 1.0 + avg_reward
            target = max(floor, min(ceiling, target))
            self.scoring_term_weights[term] = current + lr * (target - current)

    def get_scoring_weight(self, term: str) -> float:
        """Get the current adaptive weight for a scoring term."""
        return self.scoring_term_weights.get(term, 1.0)

    def detect_obstacle_clusters(self, scan: Dict[int, float]):
        """Detect dense obstacle clusters — regions where multiple angles are blocked tightly.
        Helps the car recognize 'tight spots' vs 'sparse obstacles'."""
        if not scan:
            self.obstacle_cluster_detected = False
            return
        blocked = 0
        total = 0
        distances = []
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            if dist > 0.0:
                total += 1
                distances.append(dist)
                if dist < self.config.caution_distance:
                    blocked += 1
        if total < 2:
            self.obstacle_cluster_detected = False
            return
        blocked_ratio = blocked / total
        avg_dist = sum(distances) / len(distances) if distances else 0.0
        cluster_score = blocked_ratio * (1.0 - avg_dist / self.config.cruise_distance)
        self.obstacle_cluster_history.append(cluster_score)
        is_cluster = (blocked_ratio >= self.config.obstacle_cluster_blocked_ratio
                      and avg_dist < self.config.cruise_distance)
        self.obstacle_cluster_detected = is_cluster
        self.obstacle_cluster_tightness = min(1.0, max(0.0,
            (self.config.cruise_distance - avg_dist) / self.config.cruise_distance
        )) if is_cluster else 0.0

    def get_cluster_caution_multiplier(self) -> float:
        """Extra caution multiplier when in a tight obstacle cluster."""
        if not self.obstacle_cluster_detected:
            return 1.0
        return 1.0 + (self.config.cluster_caution_multiplier - 1.0) * self.obstacle_cluster_tightness

    def get_cluster_speed_multiplier(self) -> float:
        """Speed reduction in tight obstacle clusters."""
        if not self.obstacle_cluster_detected:
            return 1.0
        return 1.0 - self.config.cluster_speed_reduction * self.obstacle_cluster_tightness

    def evaluate_flow_state(self, front_distance: float, mode: str):
        """Track whether the car is in a 'flow state' (smooth navigation) or 'struggling'.
        Uses recent decision quality: good = front distance improving, bad = escapes/dead-ends."""
        was_good = front_distance > self.config.caution_distance and mode == "drive"
        self.flow_state_history.append(was_good)
        if len(self.flow_state_history) < 5:
            return
        good_ratio = sum(self.flow_state_history) / len(self.flow_state_history)
        self.flow_state_ratio = good_ratio
        if good_ratio >= self.config.flow_state_success_threshold:
            self.flow_state = "flow"
        elif good_ratio <= self.config.flow_state_struggle_threshold:
            self.flow_state = "struggle"
        else:
            self.flow_state = "normal"

    def get_flow_state_speed_bonus(self) -> float:
        """Small speed boost when in flow state."""
        if self.flow_state == "flow":
            return self.config.flow_state_speed_bonus * self.flow_state_ratio
        return 0.0

    def get_flow_state_caution_multiplier(self) -> float:
        """Reduced caution when in flow (confidence boost)."""
        if self.flow_state == "flow":
            return self.config.flow_state_caution_reduction
        if self.flow_state == "struggle":
            return self.config.struggle_caution_multiplier
        return 1.0

    def generate_fortune(self, now: float) -> str:
        """Generate a fortune-teller style prediction based on recent navigation patterns."""
        if now - self.last_fortune_time < self.config.fortune_teller_cooldown:
            return ""
        if len(self.flow_state_history) < 5:
            return ""
        personality = self.get_personality_label()
        candidates = []
        good_ratio = self.flow_state_ratio
        if good_ratio > 0.7:
            candidates.append(f"the stars align: {personality} sees smooth sailing ahead")
            candidates.append(f"my sensors predict open roads and good vibes")
            candidates.append(f"fortune says: keep going, the best path is yet to come")
        elif good_ratio < 0.3:
            candidates.append(f"the crystals are cloudy... {personality} senses trouble ahead")
            candidates.append(f"fortune cookie: 'the wall you seek is the wall that finds you'")
            candidates.append(f"my inner oracle says: brace for impact, {personality}")
        if self.obstacle_cluster_detected:
            candidates.append(f"I foresee a tight spot coming... {personality} should stay sharp")
            candidates.append(f"the maze whispers: obstacles cluster, but so does wisdom")
        if self.flow_state == "flow" and self.success_streak >= 5:
            candidates.append(f"the prophecy is clear: {personality} is unstoppable today")
            candidates.append(f"fortune: {self.success_streak}-streak and counting. the universe favors the bold")
        if self.escape_count > 0 and self.escape_count % 7 == 0:
            candidates.append(f"lucky number {self.escape_count} escapes! {personality} is due for a breakthrough")
        if self.terrain_class == "open":
            candidates.append(f"the horizon calls to {personality}: venture forth, the world is wide")
        elif self.terrain_class == "cluttered":
            candidates.append(f"the tea leaves say: chaos is just order waiting to be understood")
        if self.loop_count > 0:
            candidates.append(f"the fates have spoken: break the cycle, {personality}")
        if self.stress > 0.5:
            candidates.append(f"the spirits sense tension. {personality}, breathe. the walls aren't personal")
        if not candidates:
            return ""
        quip = candidates[int(abs(now * 2657 + self.total_loops * 19)) % len(candidates)]
        for topic in self.fortune_predictions:
            if topic in quip:
                return ""
        self.last_fortune_time = now
        self.fortune_predictions.append(quip[:40])
        return quip

    # ─── Gen18: Coverage map ───────────────────────────────────────────────

    def coverage_cell_for_angle(self, angle: int) -> int:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        normalized = (angle + max_angle) / (2.0 * max_angle)
        cell = int(normalized * self.config.coverage_map_cells)
        return max(0, min(self.config.coverage_map_cells - 1, cell))

    def update_coverage_map(self, heading: int):
        decay = self.config.coverage_map_decay
        for cell in self.coverage_map:
            self.coverage_map[cell] *= decay
        cell = self.coverage_cell_for_angle(heading)
        prev = self.coverage_map.get(cell, 0.0)
        self.coverage_map[cell] = min(1.0, prev + self.config.coverage_map_visit_gain)
        if prev < 0.1 and self.coverage_map[cell] >= 0.1:
            self.total_coverage_cells_visited += 1
        self.coverage_map_last_cell = cell

    def coverage_bonus_for_angle(self, angle: int) -> float:
        cell = self.coverage_cell_for_angle(angle)
        visits = self.coverage_map.get(cell, 0.0)
        if visits < 0.05:
            return self.config.coverage_novelty_bonus
        if visits > 0.6:
            return self.config.coverage_revisit_penalty * visits
        return 0.0

    def coverage_ratio(self) -> float:
        if self.config.coverage_map_cells == 0:
            return 0.0
        return self.total_coverage_cells_visited / self.config.coverage_map_cells

    # ─── Gen18: Frontier-based exploration ────────────────────────────────

    def frontier_cell_for_angle(self, angle: int) -> int:
        max_angle = max(abs(a) for a in self.config.scan_angles) or 1
        normalized = (angle + max_angle) / (2.0 * max_angle)
        cell = int(normalized * self.config.frontier_cell_count)
        return max(0, min(self.config.frontier_cell_count - 1, cell))

    def update_frontier_map_gen18(self, scan: Dict[int, float]):
        decay = self.config.frontier_decay
        for cell in self.frontier_map_gen18:
            self.frontier_map_gen18[cell] *= decay
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            cell = self.frontier_cell_for_angle(angle)
            if dist > self.config.frontier_open_threshold:
                self.frontier_map_gen18[cell] = max(0.0, self.frontier_map_gen18.get(cell, 0.0) - 0.2)
            elif 0 < dist < self.config.frontier_blocked_threshold:
                self.frontier_map_gen18[cell] = min(1.0, self.frontier_map_gen18.get(cell, 0.0) + 0.3)
            elif dist > 0:
                self.frontier_map_gen18[cell] = min(1.0, self.frontier_map_gen18.get(cell, 0.5) + 0.05)
        for cell in list(self.frontier_map_gen18.keys()):
            if self.frontier_map_gen18[cell] < 0.02:
                del self.frontier_map_gen18[cell]

    def frontier_bonus_for_angle_gen18(self, angle: int) -> float:
        cell = self.frontier_cell_for_angle(angle)
        frontier_score = self.frontier_map_gen18.get(cell, 0.5)
        if 0.2 < frontier_score < 0.8:
            return frontier_score * self.config.frontier_exploration_bonus
        return 0.0

    # ─── Gen18: Straight-line preference ──────────────────────────────────

    def update_straight_line_state(self, heading: int, now: float, front_distance: float):
        if abs(heading) < 15 and front_distance > self.config.caution_distance:
            if self.straight_line_heading == 0 or abs(heading - self.straight_line_heading) < 10:
                if self.straight_line_heading == 0:
                    self.straight_line_start_time = now
                self.straight_line_heading = heading
                elapsed = now - self.straight_line_start_time
                self.directional_persistence = min(
                    self.config.directional_persistence_max,
                    self.directional_persistence + self.config.directional_persistence_gain * elapsed
                )
                if elapsed > self.config.straight_line_duration_threshold:
                    self.straight_line_bonus_active = True
            else:
                self._reset_straight_line(now)
        else:
            self._reset_straight_line(now)

    def _reset_straight_line(self, now: float):
        self.straight_line_start_time = now
        self.straight_line_heading = 0
        self.straight_line_bonus_active = False
        self.directional_persistence *= self.config.directional_persistence_decay
        if self.directional_persistence < 0.01:
            self.directional_persistence = 0.0

    def straight_line_score_bonus(self, angle: int, now: float) -> float:
        if not self.straight_line_bonus_active:
            return 0.0
        if abs(angle) < 20:
            elapsed = now - self.straight_line_start_time
            bonus = self.config.straight_line_bonus + elapsed * 2.0
            return min(self.config.straight_line_max_bonus, bonus)
        return 0.0

    def straight_line_grip_correction(self, steer: float, heading: int) -> float:
        if not self.straight_line_bonus_active or abs(heading) >= 20:
            return steer
        grip = self.config.straight_line_grip_gain
        return steer * (1.0 - grip)

    # ─── Gen19+Gen20: Emotional moves ─────────────────────────────────────

    def check_emotional_triggers(self, front_distance: float, now: float, mode: str):
        if self.emotional_move_active:
            return
        self.check_discovery(front_distance, now)
        if mode == "drive" and front_distance > self.config.open_space_distance:
            if self.success_streak >= 10 and self.success_streak % 5 == 0:
                self.trigger_emotional_move("victory_dance", now)
                self.set_mood("celebration", now, "streak_milestone")
        elif mode == "drive" and front_distance > self.config.cruise_distance:
            if self.success_streak >= 5 and self.success_streak % 5 == 0:
                self.trigger_emotional_move("happy_wiggle", now)
                self.set_mood("confident", now, "good_streak")
        elif mode == "escape" and self.escape_count > 0 and self.escape_count % 7 == 0:
            self.trigger_emotional_move("frustrated_shimmy", now)
            self.set_mood("stressed", now, "escape_frustration")
        elif mode == "drive" and self.straight_line_bonus_active:
            elapsed = now - self.straight_line_start_time
            if elapsed > 4.0 and int(elapsed) % 5 == 0:
                self.trigger_emotional_move("confident_strut", now)
                self.set_mood("confident", now, "long_straight_path")
        elif mode == "drive" and front_distance > self.config.open_space_distance * 1.2:
            if self.total_loops % 100 == 0:
                self.trigger_emotional_move("curious_tilt", now)
        elif mode == "escape" and self.escape_count > 0 and self.escape_count % 3 == 0:
            if self.escape_count % 21 == 0:
                self.trigger_emotional_move("contemplative_circle", now)
            else:
                self.trigger_emotional_move("surprise_shake", now)
        if mode == "drive" and front_distance < self.config.caution_distance and front_distance > self.config.danger_distance:
            if self.total_loops % 80 == 0:
                self.trigger_emotional_move("nervous_creep", now)
                self.set_mood("stressed", now, "nervous_approach")
        elif mode == "drive" and self.success_streak >= 20 and self.success_streak % 10 == 0:
            self.trigger_emotional_move("triumphant_arc", now)
            self.set_mood("celebration", now, "triumph")
        elif mode == "drive" and self.loop_count > 0 and self.loop_count % 2 == 0 and now - self.last_loop_detection_time < 10.0:
            if self.total_loops % 60 == 0:
                self.trigger_emotional_move("confused_figure8", now)
                self.set_mood("exploring", now, "confused")
        elif mode == "drive" and front_distance > self.config.open_space_distance and self.stress < 0.1:
            if self.total_loops % 150 == 0:
                self.trigger_emotional_move("relaxed_cruise", now)
                self.set_mood("confident", now, "relaxed")
        elif mode == "drive" and self.total_loops > 0 and self.total_loops % 200 == 0:
            self.trigger_emotional_move("alert_scan", now)

        # Gen21: new emotional moves
        if mode == "drive" and self.satisfaction_level > 0.8 and self.success_streak >= 8:
            if self.total_loops % 120 == 0:
                self.trigger_emotional_move("satisfied_purr", now)
                self.set_mood("confident", now, "satisfaction")
        elif mode == "drive" and front_distance > self.config.cruise_distance and self.curiosity_drive > 0.6:
            if self.total_loops % 90 == 0:
                self.trigger_emotional_move("curious_sniff", now)
                self.set_mood("exploring", now, "curiosity")
        elif mode == "drive" and self.cruise_mode_active and self.cruise_mode_escalation >= 2:
            if self.total_loops % 80 == 0:
                self.trigger_emotional_move("gentle_weave", now)
                self.set_mood("confident", now, "cruise_flow")
        elif mode == "drive" and self.success_streak >= 25 and self.success_streak % 5 == 0:
            if self.total_loops % 60 == 0:
                self.trigger_emotional_move("victory_lap", now)
                self.set_mood("celebration", now, "mega_streak")
        elif mode == "drive" and front_distance > self.config.open_space_distance and self.total_loops % 250 == 0:
            self.trigger_emotional_move("grateful_bow", now)
            self.set_mood("celebration", now, "gratitude")
        elif mode == "drive" and front_distance > self.config.cruise_distance and self.satisfaction_level > 0.5:
            if self.total_loops % 100 == 0:
                self.trigger_emotional_move("happy_hop", now)
        elif mode == "drive" and self.dead_zone_map and self.total_loops % 110 == 0:
            self.trigger_emotional_move("determined_lunge", now)
            self.set_mood("exploring", now, "determination")

        # Gen22: new emotional moves
        if mode == "drive" and self.success_streak >= self.config.zoomies_streak_threshold:
            if self.total_loops % 70 == 0 and now - self.last_zoomies_time > 15.0:
                self.trigger_emotional_move("zoomies", now)
                self.set_mood("celebration", now, "zoomies_burst")
                self.last_zoomies_time = now
        elif mode == "drive" and front_distance > self.config.play_bow_open_space_threshold:
            if self.total_loops % 180 == 0 and self.playfulness_level > 0.5:
                self.trigger_emotional_move("play_bow", now)
                self.set_mood("confident", now, "playful")
        elif mode == "drive" and front_distance < self.config.stalk_mode_distance and front_distance > self.config.danger_distance:
            if self.total_loops % 90 == 0 and self.curiosity_drive > 0.4:
                self.trigger_emotional_move("stalk_mode", now)
                self.set_mood("exploring", now, "predatory")
        elif mode == "drive" and front_distance < self.config.greeting_distance_threshold:
            if self.total_loops % 60 == 0 and now - self.last_greeting_time > 10.0:
                self.trigger_emotional_move("greeting", now)
                self.set_mood("celebration", now, "friendly_greeting")
                self.last_greeting_time = now
        elif mode == "drive" and self.low_activity_start > 0 and (now - self.low_activity_start) > self.config.sleep_mode_inactivity_s:
            if self.total_loops % 50 == 0 and now - self.last_sleep_mode_time > 20.0:
                self.trigger_emotional_move("sleep_mode", now)
                self.set_mood("neutral", now, "resting")
                self.last_sleep_mode_time = now
                self.low_activity_start = -math.inf
        elif mode == "drive" and self.escape_count > 0 and self.escape_count % 5 == 0:
            if self.total_loops % 80 == 0:
                self.trigger_emotional_move("backing_dance", now)
                self.set_mood("stressed", now, "backing_away")
        elif mode == "drive" and self.grid_coverage_ratio() > self.config.serpentine_coverage_threshold:
            if self.total_loops % 130 == 0 and self.satisfaction_level > 0.6:
                self.trigger_emotional_move("serpentine", now)
                self.set_mood("celebration", now, "coverage_milestone")
        elif mode == "drive" and front_distance > self.config.cruise_distance and self.playfulness_level > 0.7:
            if self.total_loops % 100 == 0:
                self.trigger_emotional_move("peek_a_boo", now)
                self.set_mood("confident", now, "playful_peek")

        # Gen23: new emotional triggers
        if mode == "drive" and self.count_unvisited_nodes() > 0 and self.total_loops % 90 == 0:
            if self.topological_current_node >= 0 and self.topological_current_node not in self.topological_visited:
                self.trigger_emotional_move("explorer_pride", now)
                self.set_mood("celebration", now, "explorer_pride")
                self.topological_visited.add(self.topological_current_node)
        elif mode == "drive" and self.room_boundary_confirmed and self.total_loops % 200 == 0:
            openness = self.get_room_openness()
            if openness > 0.5:
                self.trigger_emotional_move("mapping_joy", now)
                self.set_mood("celebration", now, "mapping_complete")
        elif mode == "drive" and self.terrain_class == "corridor" and self.success_streak >= 5:
            if self.total_loops % 70 == 0:
                self.trigger_emotional_move("corridor_dance", now)
                self.set_mood("confident", now, "corridor_flow")
        elif mode == "drive" and front_distance > self.config.open_space_distance * 1.3:
            if self.total_loops % 150 == 0 and self.success_streak >= 8:
                self.trigger_emotional_move("open_space_celebration", now)
                self.set_mood("celebration", now, "open_space_joy")
        elif mode == "drive" and self.terrain_class == "corridor" and self.total_loops % 110 == 0:
            if self.wall_following_escape_active or (self.last_scan.get(-45, 0) > 0 and self.last_scan.get(45, 0) > 0):
                self.trigger_emotional_move("wall_caress", now)
                self.set_mood("exploring", now, "wall_caress")
        elif mode == "drive" and self.discovery_count > 0 and self.total_loops % 180 == 0:
            if front_distance > self.config.open_space_distance * 1.5:
                self.trigger_emotional_move("discovery_spin", now)
                self.set_mood("celebration", now, "discovery_spin")

    # ─── Gen18: Mood system ───────────────────────────────────────────────

    def set_mood(self, mood: str, now: float, trigger: str = ""):
        self.current_mood = mood
        self.mood_start_time = now
        self.celebration_trigger = trigger

    def update_mood(self, front_distance: float, now: float, mode: str):
        if self.current_mood == "celebration":
            if now - self.mood_start_time > self.config.mood_celebration_duration:
                self.current_mood = "neutral"
        elif self.current_mood == "stressed":
            if self.stress < 0.3 and now - self.mood_start_time > 3.0:
                self.current_mood = "neutral"
        elif self.current_mood == "confident":
            if front_distance < self.config.caution_distance or mode == "escape":
                self.current_mood = "neutral"
        elif self.current_mood == "exploring":
            if self.total_coverage_cells_visited > self.config.coverage_map_cells * 0.5:
                self.current_mood = "neutral"
        else:
            if self.stress > 0.5:
                self.current_mood = "stressed"
            elif self.total_coverage_cells_visited < 3 and self.total_loops > 50:
                self.current_mood = "exploring"
            elif self.success_streak >= 8 and front_distance > self.config.cruise_distance:
                self.current_mood = "confident"

    # ─── Gen19: 2D grid exploration map ───────────────────────────────────

    def heading_to_direction(self, heading_deg: float) -> Tuple[float, float]:
        rad = math.radians(heading_deg)
        return math.sin(rad), math.cos(rad)

    def update_dead_reckoning(self, front_distance: float, now: float):
        dx, dy = self.heading_to_direction(self.estimated_heading_deg)
        if self.current_speed > 0.05 and front_distance > self.config.danger_distance:
            travel = self.current_speed * self.config.loop_delay_s * 50.0
            self.estimated_x += dx * travel
            self.estimated_y += dy * travel

    def update_exploration_grid(self, now: float):
        decay = self.config.grid_decay
        for cell in list(self.exploration_grid.keys()):
            self.exploration_grid[cell] *= decay
            if self.exploration_grid[cell] < 0.01:
                del self.exploration_grid[cell]

        cx = int(self.estimated_x / self.config.grid_cell_size_cm)
        cy = int(self.estimated_y / self.config.grid_cell_size_cm)
        cx = max(0, min(self.config.grid_map_size - 1, cx + self.config.grid_map_size // 2))
        cy = max(0, min(self.config.grid_map_size - 1, cy + self.config.grid_map_size // 2))
        self.current_grid_cell = (cx, cy)
        prev = self.exploration_grid.get(self.current_grid_cell, 0.0)
        self.exploration_grid[self.current_grid_cell] = min(1.0, prev + 0.15)
        if prev < 0.05 and self.exploration_grid[self.current_grid_cell] >= 0.05:
            self.total_grid_cells_visited += 1

        if now - self.last_grid_frontier_update > 2.0:
            self.last_grid_frontier_update = now
            self._update_grid_frontier()

    def _update_grid_frontier(self):
        self.grid_frontier_cells.clear()
        for (cx, cy) in list(self.exploration_grid.keys()):
            if self.exploration_grid.get((cx, cy), 0.0) < 0.05:
                continue
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                nx, ny = cx + ddx, cy + ddy
                if 0 <= nx < self.config.grid_map_size and 0 <= ny < self.config.grid_map_size:
                    if self.exploration_grid.get((nx, ny), 0.0) < 0.05:
                        self.grid_frontier_cells.add((nx, ny))

    def grid_cell_for_heading(self, heading: int) -> Tuple[int, int]:
        cx, cy = self.current_grid_cell
        dx, dy = self.heading_to_direction(float(heading))
        steps = 2
        tx = int(cx + dx * steps)
        ty = int(cy + dy * steps)
        tx = max(0, min(self.config.grid_map_size - 1, tx))
        ty = max(0, min(self.config.grid_map_size - 1, ty))
        return (tx, ty)

    def grid_novelty_bonus_for_angle(self, angle: int) -> float:
        target_cell = self.grid_cell_for_heading(angle)
        visits = self.exploration_grid.get(target_cell, 0.0)
        if visits < 0.05:
            return self.config.grid_novelty_bonus
        if visits > 0.5:
            return self.config.grid_revisit_penalty * visits
        return 0.0

    def grid_frontier_bonus_for_angle(self, angle: int) -> float:
        if not self.grid_frontier_cells:
            return 0.0
        target_cell = self.grid_cell_for_heading(angle)
        if target_cell in self.grid_frontier_cells:
            return self.config.grid_frontier_bonus
        best_dist = float('inf')
        for fc in self.grid_frontier_cells:
            d = abs(fc[0] - target_cell[0]) + abs(fc[1] - target_cell[1])
            if d < best_dist:
                best_dist = d
        if best_dist <= 2:
            return self.config.grid_frontier_bonus * 0.5 * (1.0 - best_dist / 3.0)
        return 0.0

    def grid_coverage_ratio(self) -> float:
        total = self.config.grid_map_size * self.config.grid_map_size
        return self.total_grid_cells_visited / max(1, total)

    # ─── Gen22: boustrophedon coverage path planning ──────────────────────

    def should_activate_boustrophedon(self, now: float) -> bool:
        if self.boustrophedon_active:
            return True
        if now - self.boustrophedon_last_run_time < self.config.boustrophedon_cooldown_s:
            return False
        coverage = self.grid_coverage_ratio()
        return coverage < self.config.boustrophedon_activation_coverage and self.total_grid_cells_visited >= 3

    def update_boustrophedon(self, front_distance: float, now: float) -> bool:
        if not self.boustrophedon_active:
            return False
        elapsed = now - self.boustrophedon_start_time
        if elapsed > self.config.boustrophedon_duration_s:
            self.boustrophedon_active = False
            self.boustrophedon_last_run_time = now
            return False
        if front_distance < self.config.boustrophedon_obstacle_margin:
            self.boustrophedon_active = False
            self.boustrophedon_last_run_time = now
            return False
        return True

    def get_boustrophedon_command(self, now: float) -> MotionCommand | None:
        if not self.boustrophedon_active:
            return None
        elapsed = now - self.boustrophedon_start_time
        lane_phase = (elapsed % 1.5) / 1.5
        if lane_phase < 0.3:
            turn_steer = self.config.boustrophedon_turn_rate * self.boustrophedon_lane_direction
            speed = self.config.boustrophedon_speed * 0.5
            heading = int(self.boustrophedon_lane_direction * 60 * lane_phase / 0.3)
        else:
            turn_steer = 0.0
            speed = self.config.boustrophedon_speed
            heading = 0
        rgb = hsv_to_rgb(0.25 + 0.15 * math.sin(elapsed * 3), 0.85, 0.6 + 0.3 * math.sin(elapsed * 5))
        return MotionCommand(
            "boustrophedon",
            max(-1.0, min(1.0, speed * (1.0 + turn_steer))),
            max(-1.0, min(1.0, speed * (1.0 - turn_steer))),
            heading,
            (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        )

    def start_boustrophedon(self, now: float):
        self.boustrophedon_active = True
        self.boustrophedon_start_time = now
        self.boustrophedon_lane_direction = 1 if self.total_grid_cells_visited % 2 == 0 else -1
        self.boustrophedon_current_lane += 1

    def get_boustrophedon_heading_bias(self, now: float) -> int:
        if not self.boustrophedon_active:
            return 0
        elapsed = now - self.boustrophedon_start_time
        lane_phase = (elapsed % 1.5) / 1.5
        if lane_phase < 0.3:
            return int(self.boustrophedon_lane_direction * 60 * lane_phase / 0.3)
        return 0

    # ─── Gen22: straight-line momentum system ─────────────────────────────

    def update_straight_line_momentum(self, heading: int, now: float, front_distance: float):
        if abs(heading) < 10 and front_distance > self.config.cruise_distance:
            if self.straight_line_momentum_heading == 0 or abs(heading - self.straight_line_momentum_heading) < 15:
                elapsed = now - self.straight_line_momentum_start
                if elapsed > self.config.directional_inertia_threshold:
                    gain = self.config.straight_line_momentum_gain
                    self.straight_line_momentum = min(self.config.straight_line_momentum_max,
                                                      self.straight_line_momentum + gain * 0.06)
                else:
                    self.straight_line_momentum_start = now
                self.straight_line_momentum_heading = heading
            else:
                self.straight_line_momentum *= self.config.straight_line_momentum_decay
                self.straight_line_momentum_heading = heading
                self.straight_line_momentum_start = now
        else:
            self.straight_line_momentum *= self.config.straight_line_momentum_decay
            if abs(heading) >= 10:
                self.straight_line_momentum_heading = 0
                self.straight_line_momentum_start = now

    def get_straight_line_momentum_bonus(self, angle: int) -> float:
        if self.straight_line_momentum < 0.1:
            return 0.0
        if abs(angle) < 10:
            return self.straight_line_momentum * 12.0
        turn_penalty = abs(angle) / 10.0 * self.config.straight_line_momentum_resistance
        return -turn_penalty * self.straight_line_momentum

    def get_straight_line_momentum_speed_boost(self) -> float:
        return self.straight_line_momentum * self.config.straight_line_momentum_speed_boost

    def apply_straight_line_momentum_correction(self, steer: float, heading: int) -> float:
        if self.straight_line_momentum < 0.2:
            return steer
        inertia = self.config.directional_inertia_strength * self.straight_line_momentum
        if abs(heading) < 15:
            steer *= (1.0 - inertia)
        return steer

    # ─── Gen22: room boundary detection ───────────────────────────────────

    def update_room_shape_map(self, scan: Dict[int, float], now: float):
        decay = self.config.room_shape_decay
        for angle in list(self.room_shape_map.keys()):
            self.room_shape_map[angle] *= decay
            if self.room_shape_map[angle] < 0.01:
                del self.room_shape_map[angle]
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            cell = int(angle / (360.0 / self.config.room_shape_cells))
            cell = cell % self.config.room_shape_cells
            prev = self.room_shape_map.get(cell, 0.0)
            self.room_shape_map[cell] = max(prev, dist / self.config.room_boundary_distance)
        self.boundary_scan_history.append(dict(scan))
        if len(self.boundary_scan_history) >= self.config.room_boundary_window:
            self._confirm_room_boundaries()

    def _confirm_room_boundaries(self):
        if not self.boundary_scan_history:
            return
        avg_scan = {}
        for scan in self.boundary_scan_history:
            for angle, dist in scan.items():
                avg_scan[angle] = avg_scan.get(angle, 0.0) + dist
        for angle in avg_scan:
            avg_scan[angle] /= len(self.boundary_scan_history)
        boundary_count = sum(1 for d in avg_scan.values() if d > self.config.room_boundary_distance)
        self.room_boundary_confirmed = boundary_count >= 3

    def get_room_openness(self) -> float:
        if not self.room_shape_map:
            return 0.0
        return sum(self.room_shape_map.values()) / len(self.room_shape_map)

    # ─── Gen22: enhanced emotional state tracking ─────────────────────────

    def update_playfulness(self, front_distance: float, now: float, mode: str):
        if mode == "drive" and front_distance > self.config.open_space_distance:
            self.playfulness_level = min(1.0, self.playfulness_level + 0.015)
            self.energy_level = min(1.0, self.energy_level + 0.008)
        elif mode == "escape":
            self.playfulness_level = max(0.0, self.playfulness_level - 0.08)
            self.energy_level = max(0.2, self.energy_level - 0.05)
        elif front_distance < self.config.caution_distance:
            self.playfulness_level = max(0.0, self.playfulness_level - 0.02)
        else:
            self.playfulness_level = max(0.0, self.playfulness_level - 0.005)
            self.energy_level = max(0.3, self.energy_level - 0.003)

    def update_activity_state(self, front_distance: float, speed: float, now: float):
        if speed < 0.1 and front_distance > self.config.danger_distance:
            if self.low_activity_start < 0:
                self.low_activity_start = now
        else:
            self.low_activity_start = -math.inf

    # ─── Gen23: Topological room graph ────────────────────────────────────

    def classify_current_area(self, front_distance: float, scan: Dict[int, float]) -> str:
        left = self.sanitize_distance(scan.get(-45, 0.0))
        right = self.sanitize_distance(scan.get(45, 0.0))
        side_sum = left + right if left > 0 and right > 0 else 0
        if front_distance > self.config.topological_room_open_threshold and side_sum > self.config.topological_room_open_threshold * 1.5:
            return "room"
        if side_sum > 0 and side_sum < self.config.topological_corridor_width_max and front_distance > self.config.caution_distance:
            return "corridor"
        if front_distance > self.config.topological_doorway_width_min and (left > self.config.topological_doorway_width_min or right > self.config.topological_doorway_width_min):
            return "doorway"
        if front_distance > self.config.cruise_distance:
            return "open"
        return "cluttered"

    def register_topological_node(self, now: float):
        if now - self.last_topological_registration < 2.0:
            return
        if len(self.topological_nodes) >= self.config.topological_node_max:
            self._prune_topological_nodes()
        cx, cy = self.current_grid_cell
        for nid, (nx, ny, ntype, salience) in self.topological_nodes.items():
            dist = ((cx - nx) ** 2 + (cy - ny) ** 2) ** 0.5
            if dist < self.config.topological_node_spacing_cm / self.config.grid_cell_size_cm:
                self.topological_nodes[nid] = (nx, ny, ntype, min(1.0, salience + 0.1))
                return
        nid = self.next_topological_id
        self.next_topological_id += 1
        area_type = self.last_area_classification
        self.topological_nodes[nid] = (float(cx), float(cy), area_type, 1.0)
        if self.topological_current_node >= 0:
            edge = (self.topological_current_node, nid)
            self.topological_edges[edge] = now
            reverse = (nid, self.topological_current_node)
            self.topological_edges[reverse] = now
        self.topological_current_node = nid
        self.last_topological_registration = now

    def _prune_topological_nodes(self):
        if not self.topological_nodes:
            return
        sorted_nodes = sorted(self.topological_nodes.items(), key=lambda x: x[1][3])
        to_remove = max(1, len(sorted_nodes) // 4)
        for nid, _ in sorted_nodes[:to_remove]:
            del self.topological_nodes[nid]
            if nid in self.topological_visited:
                self.topological_visited.discard(nid)
            edges_to_remove = [e for e in self.topological_edges if e[0] == nid or e[1] == nid]
            for e in edges_to_remove:
                del self.topological_edges[e]

    def decay_topological_nodes(self):
        decay = self.config.topological_node_decay
        for nid in list(self.topological_nodes.keys()):
            x, y, ntype, salience = self.topological_nodes[nid]
            new_salience = salience * decay
            if new_salience < 0.02:
                del self.topological_nodes[nid]
                self.topological_visited.discard(nid)
            else:
                self.topological_nodes[nid] = (x, y, ntype, new_salience)
        edge_cutoff = time.monotonic() - self.config.topological_edge_expiry_s
        for edge in list(self.topological_edges.keys()):
            if self.topological_edges[edge] < edge_cutoff:
                del self.topological_edges[edge]

    def topological_bonus_for_heading(self, angle: int) -> float:
        if not self.topological_nodes:
            return 0.0
        cx, cy = self.current_grid_cell
        dx_dir, dy_dir = self.heading_to_direction(float(angle))
        best_bonus = 0.0
        for nid, (nx, ny, ntype, salience) in self.topological_nodes.items():
            tdx = nx - cx
            tdy = ny - cy
            dist = (tdx ** 2 + tdy ** 2) ** 0.5
            if dist < 1.0:
                continue
            target_heading = math.degrees(math.atan2(tdx, tdy))
            heading_diff = abs(angle - target_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            if heading_diff < 30:
                if nid not in self.topological_visited:
                    bonus = self.config.topological_unvisited_bonus * salience / max(1.0, dist)
                elif ntype == "room":
                    bonus = self.config.topological_frontier_bonus * salience / max(1.0, dist)
                else:
                    bonus = self.config.topological_visited_penalty * salience
                if bonus > best_bonus:
                    best_bonus = bonus
        return best_bonus

    def count_unvisited_nodes(self) -> int:
        return sum(1 for nid in self.topological_nodes if nid not in self.topological_visited)

    # ─── Gen23: VFH-inspired polar histogram ──────────────────────────────

    def build_vfh_histogram(self, scan: Dict[int, float], now: float):
        if now - self.last_vfh_update < 0.3:
            return
        self.last_vfh_update = now
        decay = self.config.vfh_histogram_decay
        for sector in list(self.vfh_histogram.keys()):
            self.vfh_histogram[sector] *= decay
            if self.vfh_histogram[sector] < 0.02:
                del self.vfh_histogram[sector]
        sector_size = 360.0 / self.config.vfh_sector_count
        for angle, dist in scan.items():
            dist = self.sanitize_distance(dist)
            if dist <= 0.0:
                continue
            for offset in range(-2, 3):
                effective_angle = angle + offset * 5.0
                sector = int((effective_angle + 180.0) / sector_size) % self.config.vfh_sector_count
                obstacle_value = max(0.0, 1.0 - dist / self.config.vfh_threshold_distance)
                self.vfh_histogram[sector] = min(1.0, self.vfh_histogram.get(sector, 0.0) + obstacle_value * 0.5)
        self._find_vfh_valleys()

    def _find_vfh_valleys(self):
        self.vfh_valleys.clear()
        sector_size = 360.0 / self.config.vfh_sector_count
        threshold = 0.3
        in_valley = False
        valley_start = 0
        for sector in range(self.config.vfh_sector_count):
            density = self.vfh_histogram.get(sector, 0.0)
            if density < threshold and not in_valley:
                in_valley = True
                valley_start = sector
            elif (density >= threshold or sector == self.config.vfh_sector_count - 1) and in_valley:
                in_valley = False
                valley_end = sector if density >= threshold else sector + 1
                width = valley_end - valley_start
                if width >= self.config.vfh_valley_min_width:
                    center_sector = (valley_start + valley_end) / 2.0
                    center_angle = center_sector * sector_size - 180.0
                    center_angle = max(-80, min(80, center_angle))
                    self.vfh_valleys.append((valley_start, valley_end, center_angle, width))

    def vfh_bonus_for_heading(self, angle: int) -> float:
        if not self.vfh_valleys:
            return 0.0
        best_bonus = 0.0
        for start_s, end_s, center_angle, width in self.vfh_valleys:
            angle_diff = abs(angle - center_angle)
            if angle_diff < 30:
                bonus = self.config.vfh_valley_center_bonus * (1.0 - angle_diff / 30.0)
                bonus += width * self.config.vfh_valley_width_bonus
                if bonus > best_bonus:
                    best_bonus = bonus
        return best_bonus

    # ─── Gen23: enhanced exploration intelligence ─────────────────────────

    def register_open_space_beacon(self, heading: int, front_distance: float, now: float):
        if len(self.open_space_beacons) >= self.config.exploration_beacon_max:
            oldest = min(self.open_space_beacons, key=lambda k: self.open_space_beacons[k][2])
            del self.open_space_beacons[oldest]
        bid = self.next_beacon_id
        self.next_beacon_id += 1
        cx, cy = self.current_grid_cell
        self.open_space_beacons[bid] = (float(heading), float(cx), float(cy))

    def decay_open_space_beacons(self, now: float):
        cutoff = now - self.config.exploration_return_expiry_s
        for bid in list(self.open_space_beacons.keys()):
            _, _, t = self.open_space_beacons[bid]
            if t < cutoff:
                del self.open_space_beacons[bid]

    def exploration_return_bonus_for_heading(self, angle: int) -> float:
        if not self.open_space_beacons:
            return 0.0
        cx, cy = self.current_grid_cell
        dx_dir, dy_dir = self.heading_to_direction(float(angle))
        best_bonus = 0.0
        for bid, (bheading, bx, by) in self.open_space_beacons.items():
            tdx = bx - cx
            tdy = by - cy
            dist = (tdx ** 2 + tdy ** 2) ** 0.5
            if dist < 2.0:
                continue
            target_heading = math.degrees(math.atan2(tdx, tdy))
            heading_diff = abs(angle - target_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            if heading_diff < 25:
                bonus = self.config.exploration_return_to_open_bonus / max(1.0, dist)
                if bonus > best_bonus:
                    best_bonus = bonus
        return best_bonus

    def record_exploration_direction(self, heading: int):
        if heading != 0:
            self.exploration_direction_history.append(heading)

    def exploration_direction_penalty(self, angle: int) -> float:
        if not self.exploration_direction_history:
            return 0.0
        penalty = 0.0
        for past_heading in self.exploration_direction_history:
            if abs(angle - past_heading) < 15:
                penalty += 1.0
        if penalty > 0:
            return penalty * self.config.exploration_direction_repeat_penalty / len(self.exploration_direction_history)
        return 0.0

    # ─── Gen19: systematic spiral exploration ─────────────────────────────

    def maybe_start_spiral_exploration(self, heading: int, front_distance: float, now: float) -> int:
        if self.spiral_active:
            elapsed = now - self.spiral_start_time
            if elapsed > self.config.spiral_duration:
                self.spiral_active = False
                self.last_spiral_time = now
                return heading
            self.spiral_angle += self.config.spiral_turn_rate
            turn_amount = math.sin(self.spiral_angle) * 30.0
            return int(heading + turn_amount)

        if now - self.last_spiral_time < self.config.spiral_exploration_interval:
            return heading
        if front_distance < self.config.cruise_distance:
            return heading
        if self.total_grid_cells_visited < 4:
            return heading
        coverage = self.grid_coverage_ratio()
        if coverage < 0.3:
            self.spiral_active = True
            self.spiral_start_time = now
            self.spiral_angle = 0.0
        return heading

    def get_spiral_command(self, now: float) -> MotionCommand | None:
        if not self.spiral_active:
            return None
        speed = self.config.spiral_speed
        turn = math.sin(self.spiral_angle) * 0.3
        left_s = speed * (1.0 + turn)
        right_s = speed * (1.0 - turn)
        hue = (now * 0.3) % 1.0
        rgb = hsv_to_rgb(hue, 0.8, 0.7)
        return MotionCommand(
            "spiral_exploration",
            max(-1.0, min(1.0, left_s)),
            max(-1.0, min(1.0, right_s)),
            int(math.sin(self.spiral_angle) * 30),
            (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
        )

    # ─── Gen19: path memory ───────────────────────────────────────────────

    def update_path_memory(self, heading: int):
        self.path_heading_history.append(heading)
        if len(self.path_heading_history) < 3:
            return
        history = list(self.path_heading_history)
        total_change = 0.0
        for i in range(1, len(history)):
            total_change += abs(history[i] - history[i - 1])
        avg_change = total_change / (len(history) - 1)
        self.path_smoothness_score = max(0.0, 1.0 - avg_change / 40.0)

    def path_smoothness_bonus_for_angle(self, angle: int) -> float:
        if self.path_smoothness_score < 0.3:
            return 0.0
        if len(self.path_heading_history) < 2:
            return 0.0
        recent = list(self.path_heading_history)[-3:]
        avg_recent = sum(recent) / len(recent)
        diff = abs(angle - avg_recent)
        if diff < 15:
            return self.config.path_smoothness_bonus * self.path_smoothness_score
        return 0.0

    # ─── Gen19: discovery excitement system ───────────────────────────────

    def check_discovery(self, front_distance: float, now: float):
        if now - self.last_discovery_time < self.config.discovery_cooldown_s:
            return
        if front_distance > self.config.open_space_distance:
            if front_distance > self.largest_open_space_seen * 1.3:
                self.largest_open_space_seen = front_distance
                self.last_discovery_time = now
                self.discovery_count += 1
                self.discovery_light_end = now + self.config.discovery_light_duration
                self.trigger_emotional_move("excited_bounce", now)
                self.set_mood("celebration", now, "discovery")

    # ─── Gen19: enhanced emotional moves ──────────────────────────────────

    def trigger_emotional_move(self, move_type: str, now: float):
        if now - self.last_emotional_move < self.config.emotional_move_cooldown:
            return
        self.emotional_move_active = True
        self.emotional_move_start = now
        self.emotional_move_type = move_type
        self.emotional_move_step = 0
        self.last_emotional_move = now

    def get_emotional_move_command(self, now: float) -> MotionCommand | None:
        if not self.emotional_move_active:
            return None
        elapsed = now - self.emotional_move_start
        move_type = self.emotional_move_type

        if move_type == "victory_dance":
            dur = self.config.victory_dance_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            wiggles = self.config.victory_dance_wiggles
            phase = (elapsed / dur) * wiggles * 2
            direction = 1 if int(phase) % 2 == 0 else -1
            speed = 0.35 * (1.0 - elapsed / dur)
            heading = 30 if direction > 0 else -30
            rgb = hsv_to_rgb(0.12 + 0.05 * math.sin(elapsed * 10), 0.9, 0.7 + 0.3 * math.sin(elapsed * 8))
            return MotionCommand("victory_dance", speed * (1.0 + direction * 0.3), speed * (1.0 - direction * 0.3), heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "happy_wiggle":
            dur = self.config.happy_wiggle_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            phase = int((elapsed / dur) * 4)
            direction = 1 if phase % 2 == 0 else -1
            speed = 0.25
            heading = 20 if direction > 0 else -20
            rgb = hsv_to_rgb(0.35 + 0.1 * math.sin(elapsed * 6), 0.8, 0.6 + 0.4 * math.sin(elapsed * 6))
            return MotionCommand("happy_wiggle", speed * (1.0 + direction * 0.3), speed * (1.0 - direction * 0.3), heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "frustrated_shimmy":
            dur = self.config.frustrated_shimmy_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            phase = int((elapsed / dur) * 8)
            direction = 1 if phase % 2 == 0 else -1
            speed = 0.15
            heading = 15 if direction > 0 else -15
            pulse = 0.5 + 0.5 * math.sin(elapsed * 20.0)
            return MotionCommand("frustrated_shimmy", speed * (1.0 + direction * 0.3), speed * (1.0 - direction * 0.3), heading,
                                 (int(180 + 75 * pulse), int(40 * (1.0 - pulse)), 0))

        elif move_type == "confident_strut":
            dur = self.config.confident_strut_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            speed = self.config.confident_strut_speed * (0.5 + 0.5 * progress)
            hue = 0.15 + 0.15 * progress
            rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * progress)
            return MotionCommand(
                "confident_strut", speed, speed, 0,
                (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            )

        elif move_type == "curious_tilt":
            dur = self.config.curious_tilt_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            phase = (elapsed / dur) * math.pi
            tilt = math.sin(phase) * 15.0
            speed = 0.2
            rgb = hsv_to_rgb(0.55, 0.7, 0.5 + 0.3 * math.sin(elapsed * 5))
            return MotionCommand("curious_tilt", speed, speed, int(tilt),
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "excited_bounce":
            dur = self.config.excited_bounce_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            phase = (elapsed / dur) * math.pi * 4
            bounce = math.sin(phase) * 0.2
            speed = 0.3 + bounce
            heading = int(math.sin(phase * 0.5) * 10)
            rgb = hsv_to_rgb(0.1 + 0.15 * math.sin(elapsed * 8), 0.9, 0.8)
            return MotionCommand("excited_bounce",
                                 max(-1.0, min(1.0, speed * 1.1)),
                                 max(-1.0, min(1.0, speed * 0.9)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "contemplative_circle":
            dur = self.config.contemplative_circle_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            angle = progress * math.pi * 2
            speed = 0.25
            left_s = speed * (1.0 + 0.4 * math.sin(angle))
            right_s = speed * (1.0 - 0.4 * math.sin(angle))
            rgb = hsv_to_rgb(0.75 + 0.1 * math.sin(elapsed * 3), 0.6, 0.4 + 0.3 * progress)
            return MotionCommand("contemplative_circle",
                                 max(-1.0, min(1.0, left_s)),
                                 max(-1.0, min(1.0, right_s)),
                                 int(math.sin(angle) * 20),
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "celebration_spin":
            dur = self.config.celebration_spin_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            speed = 0.4 * progress
            direction = 1 if self.discovery_count % 2 == 0 else -1
            rgb = hsv_to_rgb((elapsed * 2) % 1.0, 0.9, 0.7 + 0.3 * math.sin(elapsed * 12))
            return MotionCommand("celebration_spin",
                                 speed if direction > 0 else -speed,
                                 -speed if direction > 0 else speed,
                                 int(direction * 60 * progress),
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "surprise_shake":
            dur = self.config.surprise_shake_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            phase = int((elapsed / dur) * 12)
            direction = 1 if phase % 2 == 0 else -1
            speed = 0.2 * (1.0 - elapsed / dur)
            heading = direction * 25
            pulse = 0.5 + 0.5 * math.sin(elapsed * 30.0)
            return MotionCommand("surprise_shake",
                                 speed * (1.0 + direction * 0.4),
                                 speed * (1.0 - direction * 0.4),
                                 heading,
                                 (int(255 * pulse), int(200 * (1.0 - pulse)), 0))

        elif move_type == "nervous_creep":
            dur = self.config.nervous_creep_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            jitter = math.sin(elapsed * 25.0) * 0.08
            speed = 0.15 * (0.5 + 0.5 * progress) + jitter
            heading = int(math.sin(elapsed * 15.0) * 12)
            pulse = 0.5 + 0.5 * math.sin(elapsed * 8.0)
            rgb = hsv_to_rgb(0.05 + 0.05 * pulse, 0.9, 0.4 + 0.3 * pulse)
            return MotionCommand("nervous_creep",
                                 max(-1.0, min(1.0, speed * (1.0 + jitter * 3))),
                                 max(-1.0, min(1.0, speed * (1.0 - jitter * 3))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "triumphant_arc":
            dur = self.config.triumphant_arc_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            arc = math.sin(progress * math.pi) * 0.5
            speed = 0.5 * (0.5 + 0.5 * progress)
            left_s = speed * (1.0 + arc)
            right_s = speed * (1.0 - arc)
            heading = int(arc * 40)
            hue = 0.1 + 0.08 * math.sin(progress * math.pi * 3)
            rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * math.sin(elapsed * 6))
            return MotionCommand("triumphant_arc",
                                 max(-1.0, min(1.0, left_s)),
                                 max(-1.0, min(1.0, right_s)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "confused_figure8":
            dur = self.config.confused_figure8_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            angle = progress * math.pi * 4
            speed = 0.2
            left_s = speed * (1.0 + 0.5 * math.sin(angle))
            right_s = speed * (1.0 - 0.5 * math.sin(angle * 0.7))
            heading = int(math.sin(angle) * 25)
            hue = 0.7 + 0.15 * math.sin(elapsed * 4)
            rgb = hsv_to_rgb(hue, 0.5, 0.4 + 0.3 * math.sin(elapsed * 3))
            return MotionCommand("confused_figure8",
                                 max(-1.0, min(1.0, left_s)),
                                 max(-1.0, min(1.0, right_s)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "relaxed_cruise":
            dur = self.config.relaxed_cruise_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            speed = 0.35 * (0.7 + 0.3 * math.sin(progress * math.pi))
            heading = int(math.sin(elapsed * 2) * 5)
            hue = 0.35 + 0.1 * math.sin(elapsed * 1.5)
            rgb = hsv_to_rgb(hue, 0.6, 0.5 + 0.3 * math.sin(elapsed * 2))
            return MotionCommand("relaxed_cruise",
                                 speed, speed, heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "alert_scan":
            dur = self.config.alert_scan_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            sweep = math.sin(progress * math.pi * 2) * 0.15
            speed = 0.1
            heading = int(math.sin(progress * math.pi * 2) * 40)
            pulse = 0.5 + 0.5 * math.sin(elapsed * 12.0)
            rgb = hsv_to_rgb(0.15, 0.9, 0.4 + 0.4 * pulse)
            return MotionCommand("alert_scan",
                                 max(-1.0, min(1.0, speed * (1.0 + sweep))),
                                 max(-1.0, min(1.0, speed * (1.0 - sweep))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        # Gen21: new emotional moves
        elif move_type == "satisfied_purr":
            dur = self.config.satisfied_purr_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            purr = math.sin(elapsed * 20.0) * 0.03
            speed = 0.3 + purr
            heading = int(math.sin(elapsed * 3.0) * 5)
            warm = 0.1 + 0.05 * math.sin(elapsed * 4.0)
            rgb = hsv_to_rgb(warm, 0.7, 0.5 + 0.3 * math.sin(elapsed * 6.0))
            return MotionCommand("satisfied_purr",
                                 speed, speed, heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "determined_lunge":
            dur = self.config.determined_lunge_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            accel = progress * 0.6
            speed = 0.2 + accel
            heading = int(math.sin(progress * math.pi) * 10)
            rgb = hsv_to_rgb(0.08 + 0.05 * progress, 0.9, 0.4 + 0.5 * progress)
            return MotionCommand("determined_lunge",
                                 max(-1.0, min(1.0, speed * 1.1)),
                                 max(-1.0, min(1.0, speed * 0.9)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "grateful_bow":
            dur = self.config.grateful_bow_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            bow = math.sin(progress * math.pi) * 0.15
            speed = 0.2 * (1.0 - bow)
            heading = 0
            rgb = hsv_to_rgb(0.12, 0.8, 0.5 + 0.4 * math.sin(progress * math.pi))
            return MotionCommand("grateful_bow",
                                 speed, speed, heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "victory_lap":
            dur = self.config.victory_lap_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            angle = progress * math.pi * 4
            speed = 0.4 + 0.2 * math.sin(progress * math.pi)
            left_s = speed * (1.0 + 0.3 * math.sin(angle))
            right_s = speed * (1.0 - 0.3 * math.sin(angle))
            heading = int(math.sin(angle) * 35)
            hue = (elapsed * 0.5) % 1.0
            rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * math.sin(elapsed * 8))
            return MotionCommand("victory_lap",
                                 max(-1.0, min(1.0, left_s)),
                                 max(-1.0, min(1.0, right_s)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "curious_sniff":
            dur = self.config.curious_sniff_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            sniff = math.sin(progress * math.pi * 3) * 0.08
            speed = 0.15 + sniff
            heading = int(math.sin(progress * math.pi * 2) * 20)
            rgb = hsv_to_rgb(0.55 + 0.1 * math.sin(elapsed * 3), 0.6, 0.4 + 0.3 * math.sin(elapsed * 5))
            return MotionCommand("curious_sniff",
                                 max(-1.0, min(1.0, speed * 1.1)),
                                 max(-1.0, min(1.0, speed * 0.9)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "gentle_weave":
            dur = self.config.gentle_weave_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            weave = math.sin(progress * math.pi * 2) * 0.12
            speed = 0.5 + 0.1 * math.sin(progress * math.pi)
            heading = int(weave * 30)
            rgb = hsv_to_rgb(0.3 + 0.1 * math.sin(elapsed * 2), 0.7, 0.5 + 0.3 * progress)
            return MotionCommand("gentle_weave",
                                 max(-1.0, min(1.0, speed * (1.0 + weave))),
                                 max(-1.0, min(1.0, speed * (1.0 - weave))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "happy_hop":
            dur = self.config.happy_hop_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            hop = math.sin(progress * math.pi * 4) * 0.15
            speed = 0.3 + hop
            heading = int(math.sin(progress * math.pi * 2) * 15)
            rgb = hsv_to_rgb(0.15 + 0.1 * math.sin(elapsed * 8), 0.85, 0.6 + 0.3 * math.sin(elapsed * 10))
            return MotionCommand("happy_hop",
                                 max(-1.0, min(1.0, speed * 1.1)),
                                 max(-1.0, min(1.0, speed * 0.9)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        # Gen22: new emotional moves
        elif move_type == "zoomies":
            dur = self.config.zoomies_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            burst = math.sin(progress * math.pi * 6) * 0.3
            speed = 0.6 + burst
            heading = int(math.sin(progress * math.pi * 8) * 20)
            hue = (elapsed * 3.0) % 1.0
            rgb = hsv_to_rgb(hue, 0.95, 0.7 + 0.3 * math.sin(elapsed * 15))
            return MotionCommand("zoomies",
                                 max(-1.0, min(1.0, speed * (1.0 + burst * 0.5))),
                                 max(-1.0, min(1.0, speed * (1.0 - burst * 0.5))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "play_bow":
            dur = self.config.play_bow_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            bow = math.sin(progress * math.pi) * 0.2
            speed = 0.15 * (1.0 - bow)
            heading = int(math.sin(progress * math.pi * 2) * 10)
            rgb = hsv_to_rgb(0.12, 0.8, 0.5 + 0.4 * math.sin(elapsed * 4))
            return MotionCommand("play_bow",
                                 max(-1.0, min(1.0, speed * 1.1)),
                                 max(-1.0, min(1.0, speed * 0.9)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "peek_a_boo":
            dur = self.config.peek_a_boo_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            peek = math.sin(progress * math.pi * 3)
            speed = 0.2 * max(0, peek)
            heading = int(peek * 30)
            rgb = hsv_to_rgb(0.55 + 0.1 * math.sin(elapsed * 6), 0.7, 0.4 + 0.4 * max(0, peek))
            return MotionCommand("peek_a_boo",
                                 max(-1.0, min(1.0, speed * 1.1)),
                                 max(-1.0, min(1.0, speed * 0.9)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "stalk_mode":
            dur = self.config.stalk_mode_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            creep = 0.05 + 0.15 * progress
            speed = creep * (0.8 + 0.2 * math.sin(elapsed * 8))
            heading = int(math.sin(elapsed * 2) * 8)
            rgb = hsv_to_rgb(0.08, 0.9, 0.3 + 0.4 * progress)
            return MotionCommand("stalk_mode",
                                 max(-1.0, min(1.0, speed * (1.0 + math.sin(elapsed * 8) * 0.1))),
                                 max(-1.0, min(1.0, speed * (1.0 - math.sin(elapsed * 8) * 0.1))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "greeting":
            dur = self.config.greeting_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            wiggle = math.sin(progress * math.pi * 4) * 0.15
            speed = 0.25 * (0.5 + 0.5 * math.sin(progress * math.pi))
            heading = int(wiggle * 25)
            hue = 0.15 + 0.1 * math.sin(elapsed * 5)
            rgb = hsv_to_rgb(hue, 0.85, 0.6 + 0.3 * math.sin(elapsed * 8))
            return MotionCommand("greeting",
                                 max(-1.0, min(1.0, speed * (1.0 + wiggle))),
                                 max(-1.0, min(1.0, speed * (1.0 - wiggle))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "sleep_mode":
            dur = self.config.sleep_mode_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            breathe = math.sin(progress * math.pi * 2) * 0.05
            speed = 0.05 + breathe
            heading = int(math.sin(progress * math.pi) * 5)
            rgb = hsv_to_rgb(0.65, 0.3, 0.15 + 0.15 * math.sin(progress * math.pi))
            return MotionCommand("sleep_mode",
                                 max(-1.0, min(1.0, speed)),
                                 max(-1.0, min(1.0, speed)),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "backing_dance":
            dur = self.config.backing_dance_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            dance = math.sin(progress * math.pi * 6)
            speed = -0.15 * (0.5 + 0.5 * math.sin(progress * math.pi))
            heading = int(dance * 20)
            hue = 0.8 + 0.1 * math.sin(elapsed * 10)
            rgb = hsv_to_rgb(hue, 0.7, 0.4 + 0.4 * math.sin(elapsed * 8))
            return MotionCommand("backing_dance",
                                 max(-1.0, min(1.0, speed * (1.0 + dance * 0.3))),
                                 max(-1.0, min(1.0, speed * (1.0 - dance * 0.3))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "serpentine":
            dur = self.config.serpentine_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            wave = math.sin(progress * math.pi * 4) * 0.2
            speed = 0.4 + 0.15 * math.sin(progress * math.pi)
            heading = int(wave * 40)
            hue = (elapsed * 0.8) % 1.0
            rgb = hsv_to_rgb(hue, 0.85, 0.6 + 0.3 * math.sin(elapsed * 6))
            return MotionCommand("serpentine",
                                 max(-1.0, min(1.0, speed * (1.0 + wave))),
                                 max(-1.0, min(1.0, speed * (1.0 - wave))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        # Gen23: new emotional moves
        elif move_type == "explorer_pride":
            dur = self.config.explorer_pride_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            arc = math.sin(progress * math.pi) * 0.3
            speed = 0.35 + 0.15 * math.sin(progress * math.pi)
            heading = int(arc * 30)
            hue = 0.15 + 0.1 * math.sin(elapsed * 4)
            rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * math.sin(elapsed * 6))
            return MotionCommand("explorer_pride",
                                 max(-1.0, min(1.0, speed * (1.0 + arc * 0.3))),
                                 max(-1.0, min(1.0, speed * (1.0 - arc * 0.3))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "mapping_joy":
            dur = self.config.mapping_joy_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            wave = math.sin(progress * math.pi * 3) * 0.15
            speed = 0.3 + 0.1 * math.sin(progress * math.pi)
            heading = int(wave * 25)
            hue = (elapsed * 0.4) % 1.0
            rgb = hsv_to_rgb(hue, 0.8, 0.5 + 0.4 * math.sin(elapsed * 5))
            return MotionCommand("mapping_joy",
                                 max(-1.0, min(1.0, speed * (1.0 + wave))),
                                 max(-1.0, min(1.0, speed * (1.0 - wave))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "corridor_dance":
            dur = self.config.corridor_dance_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            dance = math.sin(progress * math.pi * 4) * 0.2
            speed = 0.4 + 0.15 * math.sin(progress * math.pi)
            heading = int(dance * 35)
            hue = 0.3 + 0.15 * math.sin(elapsed * 6)
            rgb = hsv_to_rgb(hue, 0.85, 0.6 + 0.3 * math.sin(elapsed * 8))
            return MotionCommand("corridor_dance",
                                 max(-1.0, min(1.0, speed * (1.0 + dance))),
                                 max(-1.0, min(1.0, speed * (1.0 - dance))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "open_space_celebration":
            dur = self.config.open_space_celebration_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            burst = math.sin(progress * math.pi * 5) * 0.25
            speed = 0.5 + 0.2 * math.sin(progress * math.pi)
            heading = int(burst * 25)
            hue = (elapsed * 2.0) % 1.0
            rgb = hsv_to_rgb(hue, 0.95, 0.7 + 0.3 * math.sin(elapsed * 12))
            return MotionCommand("open_space_celebration",
                                 max(-1.0, min(1.0, speed * (1.0 + burst * 0.4))),
                                 max(-1.0, min(1.0, speed * (1.0 - burst * 0.4))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "wall_caress":
            dur = self.config.wall_caress_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            caress = math.sin(progress * math.pi * 2) * 0.08
            speed = 0.3 + 0.05 * math.sin(progress * math.pi)
            heading = int(caress * 20)
            hue = 0.25 + 0.08 * math.sin(elapsed * 3)
            rgb = hsv_to_rgb(hue, 0.7, 0.5 + 0.3 * math.sin(elapsed * 4))
            return MotionCommand("wall_caress",
                                 max(-1.0, min(1.0, speed * (1.0 + caress))),
                                 max(-1.0, min(1.0, speed * (1.0 - caress))),
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        elif move_type == "discovery_spin":
            dur = self.config.discovery_spin_duration
            if elapsed >= dur:
                self.emotional_move_active = False
                return None
            progress = elapsed / dur
            direction = 1 if self.discovery_count % 2 == 0 else -1
            speed = 0.35 * progress
            heading = int(direction * 80 * progress)
            hue = (elapsed * 3.0) % 1.0
            rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * math.sin(elapsed * 10))
            return MotionCommand("discovery_spin",
                                 speed * direction if direction > 0 else -speed,
                                 -speed * direction if direction > 0 else speed,
                                 heading,
                                 (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        self.emotional_move_active = False
        return None

    # ─── Gen19: enhanced mood lighting ────────────────────────────────────

    def trigger_enhanced_mood(self, effect: str, now: float):
        self.enhanced_mood_effect = effect
        self.enhanced_mood_start = now

    def get_enhanced_mood_duration(self) -> float:
        if self.enhanced_mood_effect == "aurora":
            return self.config.mood_aurora_duration
        elif self.enhanced_mood_effect == "heartbeat":
            return self.config.mood_heartbeat_duration
        elif self.enhanced_mood_effect == "rainbow_wave":
            return self.config.mood_rainbow_wave_duration
        return 0.0

    def is_enhanced_mood_active(self, now: float) -> bool:
        if not self.enhanced_mood_effect:
            return False
        return now - self.enhanced_mood_start < self.get_enhanced_mood_duration()

    def get_mood_hue_offset(self, now: float) -> float:
        if self.current_mood == "celebration":
            return (now * self.config.mood_rainbow_cycle_rate) % 1.0
        elif self.current_mood == "stressed":
            return 0.0
        elif self.current_mood == "confident":
            return 0.15
        elif self.current_mood == "exploring":
            return (now * 0.1) % 1.0
        return self.mood_hue_offset

    # ─── Gen20: Waypoint/Landmark memory ──────────────────────────────────

    def register_landmark(self, front_distance: float, now: float):
        if front_distance < self.config.landmark_detection_front_threshold:
            return
        if now - self.last_landmark_registration < 2.0:
            return
        cx, cy = self.current_grid_cell
        for lid, (lx, ly, salience) in self.landmark_memory.items():
            dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if dist < self.config.landmark_min_spacing_cm / self.config.grid_cell_size_cm:
                return
        lid = self.next_landmark_id
        self.next_landmark_id += 1
        self.landmark_memory[lid] = (float(cx), float(cy), 1.0)
        self.landmark_visit_count[lid] = 0
        self.last_landmark_registration = now

    def decay_landmarks(self):
        decay = self.config.landmark_decay
        for lid in list(self.landmark_memory.keys()):
            x, y, salience = self.landmark_memory[lid]
            new_salience = salience * decay
            if new_salience < 0.01:
                del self.landmark_memory[lid]
                if lid in self.landmark_visit_count:
                    del self.landmark_visit_count[lid]
            else:
                self.landmark_memory[lid] = (x, y, new_salience)

    def landmark_bonus_for_angle(self, angle: int) -> float:
        if not self.landmark_memory:
            return 0.0
        cx, cy = self.current_grid_cell
        dx, dy = self.heading_to_direction(float(angle))
        best_bonus = 0.0
        for lid, (lx, ly, salience) in self.landmark_memory.items():
            visits = self.landmark_visit_count.get(lid, 0)
            target_x, target_y = lx, ly
            tdx = target_x - cx
            tdy = target_y - cy
            dist = (tdx ** 2 + tdy ** 2) ** 0.5
            if dist < 0.5:
                continue
            target_heading = math.degrees(math.atan2(tdx, tdy))
            heading_diff = abs(angle - target_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            if heading_diff < 30:
                if visits == 0:
                    bonus = self.config.landmark_novelty_bonus * salience
                else:
                    bonus = self.config.landmark_visited_penalty * salience * min(visits, 3)
                if bonus > best_bonus:
                    best_bonus = bonus
        return best_bonus

    def visit_nearest_landmark(self, heading: int):
        if not self.landmark_memory:
            return
        cx, cy = self.current_grid_cell
        best_lid = None
        best_dist = float('inf')
        for lid, (lx, ly, salience) in self.landmark_memory.items():
            dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_lid = lid
        if best_lid is not None and best_dist < 2.0:
            self.landmark_visit_count[best_lid] = self.landmark_visit_count.get(best_lid, 0) + 1

    # ─── Gen20: Enhanced straight-line preference ─────────────────────────

    def update_straight_lock(self, heading: int, now: float, front_distance: float):
        if abs(heading) < 10 and front_distance > self.config.caution_distance:
            if self.straight_lock_heading == 0 or abs(heading - self.straight_lock_heading) < 8:
                if self.straight_lock_heading == 0:
                    self.straight_lock_start_time = now
                    self.straight_lock_heading = heading
                elapsed = now - self.straight_lock_start_time
                if elapsed > self.config.straight_lock_duration_threshold:
                    self.straight_lock_active = True
                if heading != 0:
                    self.last_non_zero_heading = heading
            else:
                self._reset_straight_lock(now)
        else:
            if heading != 0:
                self.last_non_zero_heading = heading
            self._reset_straight_lock(now)

    def _reset_straight_lock(self, now: float):
        if self.straight_lock_active:
            self.straight_line_recovery_active = True
        self.straight_lock_start_time = now
        self.straight_lock_heading = 0
        self.straight_lock_active = False
        self.straight_line_recovery_active = False

    def straight_lock_bonus(self, angle: int, now: float) -> float:
        if not self.straight_lock_active:
            if self.straight_line_recovery_active and abs(angle) < 15:
                return self.config.straight_line_recovery_bonus
            return 0.0
        if abs(angle) < 15:
            elapsed = now - self.straight_lock_start_time
            bonus = self.config.straight_line_heading_lock_bonus * min(1.0, elapsed / 3.0)
            return bonus
        turn_penalty = abs(angle) / 10.0 * self.config.straight_line_turn_penalty
        return -turn_penalty

    # ─── Gen20: Directed frontier exploration ─────────────────────────────

    def select_frontier_target(self, now: float):
        if now - self.last_frontier_target_time < self.config.frontier_target_cooldown:
            return
        if not self.grid_frontier_cells:
            self.frontier_directed_active = False
            return
        cx, cy = self.current_grid_cell
        best_cell = None
        best_score = -float('inf')
        for fc in self.grid_frontier_cells:
            dist = ((fc[0] - cx) ** 2 + (fc[1] - cy) ** 2) ** 0.5
            score = -dist
            for lid, (lx, ly, salience) in self.landmark_memory.items():
                landmark_dist = ((fc[0] - lx) ** 2 + (fc[1] - ly) ** 2) ** 0.5
                if landmark_dist < 3.0:
                    visits = self.landmark_visit_count.get(lid, 0)
                    if visits == 0:
                        score += 5.0 * salience
            if score > best_score:
                best_score = score
                best_cell = fc
        if best_cell is not None:
            dx = best_cell[0] - cx
            dy = best_cell[1] - cy
            target_heading = int(math.degrees(math.atan2(dx, dy)))
            target_heading = max(-80, min(80, target_heading))
            self.frontier_target_heading = target_heading
            self.frontier_directed_active = True
            self.last_frontier_target_time = now

    def frontier_directed_bonus_for_angle(self, angle: int, now: float) -> float:
        if not self.frontier_directed_active:
            return 0.0
        if now - self.last_frontier_target_time > self.config.frontier_target_cooldown * 2:
            self.frontier_directed_active = False
            return 0.0
        diff = abs(angle - self.frontier_target_heading)
        if diff < 20:
            return self.config.frontier_directed_bonus
        if diff < 45:
            return self.config.frontier_directed_bonus * (1.0 - diff / 45.0)
        return 0.0

    def update_cruise_mode(self, heading: int, front_distance: float, now: float, speed: float):
        if self.cruise_mode_active:
            if front_distance < self.config.cruise_mode_graceful_exit_distance:
                self.cruise_mode_active = False
                self.cruise_mode_escalation = 0
                return
            if now - self.cruise_mode_start_time > self.config.cruise_mode_max_duration:
                self.cruise_mode_active = False
                self.cruise_mode_escalation = 0
                return
            if abs(heading - self.cruise_mode_heading) > 25:
                self.cruise_mode_active = False
                self.cruise_mode_escalation = 0
                return
            self.cruise_mode_distance_traveled += speed * self.config.loop_delay_s * 50.0
            self.cruise_mode_escalation = min(3, int((now - self.cruise_mode_start_time) / 2.0))
        else:
            if (front_distance > self.config.cruise_mode_activation_distance
                    and abs(heading) < 10
                    and self.success_streak >= 3):
                self.cruise_mode_active = True
                self.cruise_mode_start_time = now
                self.cruise_mode_heading = heading
                self.cruise_mode_distance_traveled = 0.0
                self.cruise_mode_escalation = 0

    def apply_cruise_mode_correction(self, steer: float, heading: int, front_distance: float) -> float:
        if not self.cruise_mode_active:
            return steer
        heading_error = heading - self.cruise_mode_heading
        correction = -heading_error * self.config.cruise_mode_heading_lock_strength * 0.02
        return max(-1.0, min(1.0, steer + correction))

    def update_arc_steering(self, target_steering: float):
        self.arc_steering_target = target_steering
        diff = self.arc_steering_target - self.arc_steering_current
        if abs(diff) < 0.01:
            self.arc_steering_current = self.arc_steering_target
            self.arc_steering_rate = 0.0
            return
        if abs(self.arc_steering_rate) < self.config.arc_steering_max_rate:
            self.arc_steering_rate += math.copysign(self.config.arc_steering_accel, diff) * self.config.loop_delay_s
            self.arc_steering_rate = max(-self.config.arc_steering_max_rate,
                                         min(self.config.arc_steering_max_rate, self.arc_steering_rate))
        step = self.arc_steering_rate * self.config.loop_delay_s
        if abs(step) > abs(diff):
            step = diff
        self.arc_steering_current += step

    def apply_smooth_arc_steering(self, steer: float) -> float:
        self.update_arc_steering(steer)
        return self.arc_steering_current

    def update_satisfaction(self, front_distance: float, now: float, mode: str):
        if mode == "drive" and front_distance > self.config.cruise_distance:
            self.satisfaction_level = min(1.0, self.satisfaction_level + 0.02)
        elif mode == "escape":
            self.satisfaction_level = max(0.0, self.satisfaction_level - 0.15)
        elif front_distance < self.config.caution_distance:
            self.satisfaction_level = max(0.0, self.satisfaction_level - 0.05)
        if self.satisfaction_level > 0.7 and now - self.last_satisfaction_trigger > 10.0:
            self.last_satisfaction_trigger = now

    def update_curiosity_drive(self, front_distance: float, now: float):
        if front_distance > self.config.open_space_distance:
            self.curiosity_drive = max(0.0, self.curiosity_drive - 0.03)
        elif front_distance < self.config.caution_distance:
            self.curiosity_drive = min(1.0, self.curiosity_drive + 0.05)
        else:
            self.curiosity_drive = max(0.0, self.curiosity_drive - 0.01)

    def update_dead_zone_map(self, now: float):
        if now - self.last_dead_zone_update < 3.0:
            return
        self.last_dead_zone_update = now
        cx, cy = self.current_grid_cell
        for (dx, dy) in self.dead_zone_map:
            dist = ((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5
            if dist > 8:
                del self.dead_zone_map[(dx, dy)]
        if self.total_loops % 20 == 0 and self.current_grid_cell not in self.dead_zone_map:
            visits = self.exploration_grid.get(self.current_grid_cell, 0.0)
            if visits < 0.05:
                self.dead_zone_map[self.current_grid_cell] = 1.0

    def dead_zone_bonus_for_heading(self, angle: int) -> float:
        if not self.dead_zone_map:
            return 0.0
        cx, cy = self.current_grid_cell
        dx_dir, dy_dir = self.heading_to_direction(float(angle))
        best_bonus = 0.0
        for (dzx, dzy), salience in self.dead_zone_map.items():
            tdx = dzx - cx
            tdy = dzy - cy
            dist = (tdx ** 2 + tdy ** 2) ** 0.5
            if dist < 1.0:
                continue
            target_heading = math.degrees(math.atan2(tdx, tdy))
            heading_diff = abs(angle - target_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            if heading_diff < 30:
                bonus = 15.0 * salience / max(1.0, dist)
                if bonus > best_bonus:
                    best_bonus = bonus
        return best_bonus

    def plan(self, front_distance: float, now: float) -> MotionCommand:
        front_distance = self.observe_front(front_distance)
        approach_rate = self.track_approach_rate(front_distance, now)
        self.decay_exploration_memory()
        self.decay_danger_heatmap()
        self.decay_danger_zone()
        self.decay_flow_memory()
        self.decay_stress()
        self.update_corner_detection(front_distance, self.last_scan)
        self.update_momentum(self.current_heading)
        self.drift_history.append(self.current_heading)
        self.total_loops += 1
        self.decay_success_map()
        self.update_terrain_memory(self.last_scan)
        self.infer_obstacle_shape(front_distance, self.last_scan)
        self.update_obstacle_velocity(self.last_scan, now)
        self.detect_side_corridor(front_distance, self.last_scan, now)
        self.detect_obstacle_clusters(self.last_scan)
        self.adapt_scoring_weights()
        self.update_frontier_map_gen18(self.last_scan)
        self.update_mood(front_distance, now, "drive")

        self.update_dead_reckoning(front_distance, now)
        self.update_exploration_grid(now)
        self.update_dead_zone_map(now)
        self.update_satisfaction(front_distance, now, "drive")
        self.update_curiosity_drive(front_distance, now)

        # Gen22: room boundary detection
        self.update_room_shape_map(self.last_scan, now)

        # Gen22: emotional state tracking
        self.update_playfulness(front_distance, now, "drive")

        # Gen23: topological room graph
        area_type = self.classify_current_area(front_distance, self.last_scan)
        self.area_classification_history.append(area_type)
        if self.area_classification_history:
            counts = {}
            for a in self.area_classification_history:
                counts[a] = counts.get(a, 0) + 1
            self.last_area_classification = max(counts, key=counts.get)
        self.register_topological_node(now)
        self.decay_topological_nodes()
        if front_distance > self.config.open_space_distance:
            self.register_open_space_beacon(self.current_heading, front_distance, now)
        self.decay_open_space_beacons(now)
        self.record_exploration_direction(self.current_heading)

        # Gen23: VFH polar histogram
        self.build_vfh_histogram(self.last_scan, now)

        if self.last_escape_heading != 0 and self.prev_front_distance > 0:
            outcome = (front_distance - self.prev_front_distance) / self.prev_front_distance
            h = self.last_escape_heading
            self.escape_outcomes[h] = self.escape_outcomes.get(h, 0.0) * 0.7 + outcome * 0.3
            self.record_escape_direction_outcome(h, outcome)
            self.last_escape_heading = 0
            if outcome > 0.1:
                self.reset_recovery()
                self.navigation_insights += 1

        heading = self.select_heading(self.last_scan, front_distance, now=now)

        self.update_junction(front_distance, self.last_scan, now)
        self.record_near_miss(front_distance, heading)
        self.adapt_exploration_strategy(front_distance, self.last_scan, now)
        self.predict_openings(self.last_scan, now)
        self.detect_loop(front_distance, heading, now)

        # Gen19: check for spiral exploration trigger
        heading = self.maybe_start_spiral_exploration(heading, front_distance, now)

        # Gen22: boustrophedon coverage path planning
        if not self.boustrophedon_active and self.should_activate_boustrophedon(now):
            self.start_boustrophedon(now)
        if self.boustrophedon_active:
            if not self.update_boustrophedon(front_distance, now):
                pass
            else:
                boust_cmd = self.get_boustrophedon_command(now)
                if boust_cmd is not None:
                    self.current_speed = 0.0
                    return boust_cmd

        if self.should_trigger_patience(front_distance, self.last_scan, now):
            self.activate_patience(now)
            self.current_speed = 0.0
            return MotionCommand(
                mode="drive",
                left_speed=0.0,
                right_speed=0.0,
                heading=heading,
                colour=self.motion_colour_for("drive", heading, 0.0, front_distance),
            )

        if self.tactical_patience_active and now - self.patience_start_time >= self.config.tactical_patience_duration:
            heading = self.reassess_after_patience(self.last_scan, front_distance, now)

        if self.predict_dead_end(front_distance, self.last_scan):
            open_side = -80
            left = self.sanitize_distance(self.last_scan.get(-45, 0.0))
            right = self.sanitize_distance(self.last_scan.get(45, 0.0))
            if left > right:
                open_side = -80
            elif right > left:
                open_side = 80
            heading = int(heading * (1.0 - self.config.dead_end_prediction_avoidance) + open_side * self.config.dead_end_prediction_avoidance)

        if self.last_scan:
            for angle in list(self.last_scan.keys()):
                la_score = self.simulate_lookahead(angle, front_distance, self.last_scan, self.config.lookahead_steps)
                self.lookahead_scores[angle] = la_score

        if front_distance <= self.config.danger_distance:
            self.clear_pounce_commitment()
            self.current_speed = 0.0
            self.escalate_escape(now)
            heading = self.select_smart_escape_heading(self.last_scan, now)
            self.target_heading = 0
            self.target_heading_time = -math.inf
            self.stuck_escape_times.append(now)
            self.path_commitment_heading = 0
            self.path_commitment_time = -math.inf
            self.record_danger([0, heading], self.config.danger_score_escape)
            self.update_danger_zone([0, heading])
            self.add_stress(self.config.stress_escape_gain)
            self.corner_detected = False
            self.corner_detection_count = 0
            self.update_flow_memory(heading, front_distance)
            self.last_escape_heading = heading
            self.last_escape_front_after = front_distance
            return MotionCommand(
                mode="escape",
                left_speed=-self.config.escape_reverse_speed,
                right_speed=-self.config.escape_reverse_speed,
                heading=heading,
                colour=self.motion_colour_for("escape", heading, 0.0, front_distance),
            )

        # Brave push: if only the front is blocked and sides are clear, the obstacle
        # might be movable — charge it with decisive force before giving up.
        if (self.is_pushable_obstacle(front_distance)
                and now - self.last_push_time >= self.config.push_cooldown_s):
            self.clear_pounce_commitment()
            self.last_push_time = now
            self.current_speed = 0.0
            self.record_danger([0], self.config.danger_score_push)
            self.add_stress(self.config.stress_push_gain)
            self.corner_detected = False
            self.corner_detection_count = 0
            self.update_flow_memory(0, front_distance)
            return MotionCommand(
                mode="brave_push",
                left_speed=self.config.push_speed,
                right_speed=self.config.push_speed,
                heading=0,
                colour=self.motion_colour_for("drive", 0, self.config.push_speed, front_distance),
            )

        # Only re-trigger dead-end escape after a cooldown so the maneuver can complete
        if self.is_dead_end(front_distance) and now - self.dead_end_recovery_time >= 3.0:
            self.clear_pounce_commitment()
            self.dead_end_recovery_time = now
            self.escalate_escape(now)
            heading = self.select_smart_escape_heading(self.last_scan, now)
            self.current_speed = 0.0
            self.path_commitment_heading = 0
            self.path_commitment_time = -math.inf
            self.stuck_escape_times.append(now)
            self.record_danger([0, -45, 45], self.config.danger_score_escape)
            self.update_danger_zone([0, -45, 45])
            self.add_stress(self.config.stress_escape_gain)
            self.corner_detected = False
            self.corner_detection_count = 0
            self.update_flow_memory(heading, front_distance)
            self.last_escape_heading = heading
            self.last_escape_front_after = front_distance
            return MotionCommand(
                mode="escape",
                left_speed=-self.config.escape_reverse_speed,
                right_speed=-self.config.escape_reverse_speed,
                heading=heading,
                colour=self.motion_colour_for("escape", heading, 0.0, front_distance),
            )

        if now - self.dead_end_recovery_time < 1.5:
            self.clear_pounce_commitment()
            recovery_progress = (now - self.dead_end_recovery_time) / 1.5
            turn_speed = self.config.escape_turn_speed * max(0.6, min(1.0, recovery_progress * 2.0))
            recovery_heading = self.path_commitment_heading if self.path_commitment_heading != 0 else (-80 if sum(self.recent_turns) >= 0 else 80)
            heading = recovery_heading
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
                colour=self.motion_colour_for("dead_end_recovery", heading, 0.0, front_distance),
            )

        if self.is_stuck(now) and self.should_escalate_recovery(now):
            self.advance_recovery_stage(now)
            self.clear_pounce_commitment()
            stage = self.recovery_stage
            if stage == 1:
                wf_active, wf_cmd = self.wall_following_escape(self.last_scan, now)
                if wf_active and wf_cmd is not None:
                    self.stuck_escape_times.append(now)
                    self.add_stress(self.config.stress_escape_gain * 0.5)
                    return wf_cmd
                left_s = self.config.recovery_wiggle_speed
                right_s = -self.config.recovery_wiggle_speed
                heading = -60 if sum(self.recent_turns) >= 0 else 60
                self.note_turn(heading)
                self.stuck_escape_times.append(now)
                self.add_stress(self.config.stress_escape_gain * 0.5)
                self.record_notable_event("wiggling", now)
                return MotionCommand(
                    mode="escape",
                    left_speed=left_s,
                    right_speed=right_s,
                    heading=heading,
                    colour=self.motion_colour_for("escape", heading, 0.0, front_distance),
                )
            elif stage == 2:
                heading = self.select_smart_escape_heading(self.last_scan, now)
                self.stuck_escape_times.append(now)
                self.add_stress(self.config.stress_escape_gain)
                self.record_notable_event("reverse_spin", now)
                return MotionCommand(
                    mode="escape",
                    left_speed=-self.config.escape_reverse_speed * 1.2,
                    right_speed=-self.config.escape_reverse_speed * 1.2,
                    heading=heading,
                    colour=self.motion_colour_for("escape", heading, 0.0, front_distance),
                )
            else:
                heading = -80 if sum(self.recent_turns) >= 0 else 80
                self.stuck_escape_times.append(now)
                self.add_stress(self.config.stress_escape_gain * 1.5)
                self.record_notable_event("full_spin", now)
                return MotionCommand(
                    mode="escape",
                    left_speed=-self.config.escape_turn_speed,
                    right_speed=self.config.escape_turn_speed,
                    heading=heading,
                    colour=self.motion_colour_for("escape", heading, 0.0, front_distance),
                )

        if self.pounce_heading != 0 and now < self.pounce_until:
            return self.build_pounce_command(self.pounce_heading, front_distance, now)
        self.clear_pounce_commitment()

        if self.should_trigger_pounce(front_distance, heading):
            self.pounce_heading = heading
            self.pounce_until = now + self.config.pounce_commit_s
            return self.build_pounce_command(heading, front_distance, now)

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

        heading = self._apply_lookahead_to_heading(heading, front_distance)

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

        terrain_multiplier = self.get_terrain_speed_multiplier()
        curvature_multiplier = self.get_curvature_speed_multiplier()
        density_multiplier = self.get_density_speed_multiplier()
        confidence_multiplier = self.get_path_confidence_multiplier()
        momentum_bonus = self.get_momentum_speed_bonus()

        base_speed *= terrain_multiplier * curvature_multiplier * density_multiplier * confidence_multiplier
        base_speed = min(self.config.open_space_speed, base_speed + momentum_bonus)

        side_closing = self.detect_side_closing(self.last_scan)
        self.last_side_closing = side_closing
        if side_closing > 0.5:
            base_speed *= (1.0 - side_closing * self.config.side_closing_slowdown)

        if self.dead_end_predicted:
            base_speed *= (1.0 - self.config.dead_end_prediction_slowdown)

        if self.moving_obstacles:
            base_speed *= (1.0 / self.config.moving_obstacle_caution_multiplier)

        cluster_speed = self.get_cluster_speed_multiplier()
        base_speed *= cluster_speed

        flow_bonus = self.get_flow_state_speed_bonus()
        base_speed = min(self.config.open_space_speed, base_speed + flow_bonus)

        boldness_boost = (self.personality_boldness - 0.5) * 0.06
        base_speed = max(0.1, min(self.config.open_space_speed, base_speed + boldness_boost))

        # Gen21: cruise mode speed boost
        if self.cruise_mode_active:
            cruise_boost = self.config.cruise_mode_speed_boost * min(1.0, self.cruise_mode_escalation / 2.0)
            base_speed = min(self.config.open_space_speed, base_speed + cruise_boost)

        # Gen22: straight-line momentum speed boost
        momentum_boost = self.get_straight_line_momentum_speed_boost()
        base_speed = min(self.config.open_space_speed, base_speed + momentum_boost)

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

        curvature = self.calculate_curvature()
        if curvature > self.config.curvature_sharp_threshold and front_distance < self.config.curvature_brake_distance:
            brake_factor = 1.0 - (curvature - self.config.curvature_sharp_threshold) * 0.005
            speed *= max(0.5, brake_factor)

        self.current_speed = speed
        steer = 0.0
        max_angle = max(abs(angle) for angle in self.config.scan_angles) or 1

        self.update_straight_line_state(heading, now, front_distance)
        self.update_coverage_map(heading)
        self.check_emotional_triggers(front_distance, now, "drive")

        # Gen22: update straight-line momentum
        self.update_straight_line_momentum(heading, now, front_distance)

        # Gen21: update cruise mode
        self.update_cruise_mode(heading, front_distance, now, speed)

        # Gen20: update straight lock
        self.update_straight_lock(heading, now, front_distance)

        # Gen20: register landmarks
        self.register_landmark(front_distance, now)
        self.decay_landmarks()
        self.visit_nearest_landmark(heading)

        # Gen20: select frontier target
        self.select_frontier_target(now)

        # Gen19: update path memory
        self.update_path_memory(heading)

        if heading != 0:
            steer = max(-1.0, min(1.0, heading / max_angle))
            if front_distance <= self.config.caution_distance:
                steer *= 1.20

        # Gen21: apply smooth arc steering for gentler turns
        steer = self.apply_smooth_arc_steering(steer)

        # Gen21: cruise mode heading lock correction
        steer = self.apply_cruise_mode_correction(steer, heading, front_distance)

        # Gen22: cruise mode lane-keeping for smoother straight paths
        if self.cruise_mode_active and abs(heading) < 15:
            lane_correction = -steer * self.config.cruise_mode_lane_keeping
            steer = max(-1.0, min(1.0, steer + lane_correction))

        # Gen22: straight-line momentum correction
        steer = self.apply_straight_line_momentum_correction(steer, heading)

        if abs(heading) < 30 and front_distance > self.config.caution_distance:
            left_side = self.last_scan.get(-45, 0.0)
            right_side = self.last_scan.get(45, 0.0)
            if left_side > 0.0 and right_side > 0.0:
                total = left_side + right_side
                if total > 0.0:
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
            # Corridor narrowing: slow down when both sides are getting tighter
            min_side = min(left_side, right_side)
            if min_side < 25.0:
                corridor_squeeze = 1.0 - (min_side / 25.0)
                speed *= (1.0 - corridor_squeeze * 0.30)

        # Gap centering: when entering a gap between obstacles, actively center the car
        if front_distance < self.config.gap_entry_distance and abs(heading) < 45:
            gap_steer = self.get_gap_center_steer(left_side, right_side)
            steer = max(-1.0, min(1.0, steer + gap_steer))

        # Side obstacle avoidance: proactively steer away from nearby side obstacles
        steer = self.apply_side_obstacle_avoidance(steer, front_distance, self.last_scan)

        # Drift correction: compensate for sustained heading bias
        steer = self.apply_drift_correction(steer)

        # Corner negotiation: bias toward the open side of detected corners
        heading, speed = self.apply_corner_negotiation(heading, front_distance, speed)

        # Predictive side-collision avoidance: if moving toward a side wall, preemptively steer away
        if abs(heading) > 20 and speed > 0.3:
            target_side_dist = self.last_scan.get(heading if heading < 0 else -45 if heading < -20 else 45, 0.0)
            target_side_dist = self.sanitize_distance(target_side_dist)
            if 0 < target_side_dist < 20.0:
                avoidance_urgency = 1.0 - (target_side_dist / 20.0)
                steer -= math.copysign(avoidance_urgency * 0.4, heading)
                steer = max(-1.0, min(1.0, steer))

        # Apply temporal heading smoothing to reduce jitter
        effective_heading = self.smooth_heading(heading)

        # Exploration strategy steering
        steer = self.apply_exploration_strategy_steer(steer, front_distance, self.last_scan, now)

        # Near-miss caution: increase caution distance after clusters of close calls
        near_miss_caution = self.get_near_miss_caution_multiplier()

        # Gen22: update activity state for sleep mode detection
        self.update_activity_state(front_distance, speed, now)

        # Gen19: check for spiral exploration command
        spiral_cmd = self.get_spiral_command(now)
        if spiral_cmd is not None:
            return spiral_cmd

        emotional_cmd = self.get_emotional_move_command(now)
        if emotional_cmd is not None:
            return emotional_cmd

        left_speed = max(-1.0, min(1.0, speed * (1.0 + steer * self.config.steer_gain)))
        right_speed = max(-1.0, min(1.0, speed * (1.0 - steer * self.config.steer_gain)))

        colour = self.motion_colour_for("drive", int(round(effective_heading)), speed, front_distance)

        self.remember_heading(heading, front_distance, now)
        self.current_heading = heading
        self.record_heading_choice(heading)
        self.record_distance(speed, self.config.loop_delay_s)
        self.update_personality(front_distance, "drive")
        self.update_flow_memory(heading, front_distance)
        self.update_success_map(heading, self.prev_front_distance, front_distance)
        self.record_environmental_memory(front_distance, heading, now, "drive")
        self.evaluate_flow_state(front_distance, "drive")
        self.record_scoring_reward("chosen_heading",
            1.0 if front_distance > self.config.caution_distance else -0.5)

        return MotionCommand(
            mode="drive",
            left_speed=left_speed,
            right_speed=right_speed,
            heading=int(round(effective_heading)),
            colour=colour,
        )


def _print_scan(scan: Dict[int, float]):
    parts = "  ".join(f"{'%+d' % a}:{scan.get(a, 0):.0f}" for a in sorted(scan))
    print(f"[SCAN] {parts} cm")


def _describe_lights(command: MotionCommand, controller: "AutonomousCarController", front_distance: float) -> str:
    """Return a compact human-readable description of the current light emotion."""
    cfg = controller.config

    if command.mode == "follow":
        spd = command.left_speed
        if spd > 0.01:
            action = "chasing →"
        elif spd < -0.01:
            action = "← backing"
        else:
            action = "holding  "
        return f"feel=tracking | ALL:blue-pulse({action})"

    if command.mode == "follow_search":
        return "feel=lost     | ALL:dim-blue(searching...)"

    if command.mode == "peek_pounce":
        return "feel=locked-on | F:amber(lock) M:red(lean) R:white(thrust)"

    if command.mode == "brave_push":
        return "feel=BRAVE   | F:white(headlights!) M:gold(charge!) R:orange(thrust)"

    if command.mode == "escape":
        esc_lvl = controller.get_escape_escalation_level(time.monotonic())
        if esc_lvl == 0:
            return "feel=EEK!    | F:RED(whoops) M:dim R:AMBER(retreat!)"
        if esc_lvl == 1:
            return "feel=NOPE   | F:RED(nope!) M:dim R:AMBER(backing up...)"
        return "feel=AAAAH  | F:RED(ABORT) M:dim R:AMBER(ABORT ABORT)"

    if command.mode == "dead_end_recovery":
        return "feel=TRAPPED | ALL:magenta-pulse(spinning in existential dread)"

    if command.mode == "victory_dance":
        return "feel=ECSTATIC | ALL:gold-pulse(VICTORY DANCE!)"

    if command.mode == "happy_wiggle":
        return "feel=JOYFUL  | ALL:green-wiggle(happy wiggle~)"

    if command.mode == "frustrated_shimmy":
        return "feel=ANNOYED | ALL:red-shimmy(frustrated shimmy)"

    if command.mode == "confident_strut":
        return "feel=SWAG    | ALL:purple-glow(confident strut)"

    if command.mode == "curious_tilt":
        return "feel=CURIOUS | ALL:cyan-pulse(curious tilt~)"

    if command.mode == "excited_bounce":
        return "feel=EXCITED | ALL:yellow-bounce(discovery!)"

    if command.mode == "contemplative_circle":
        return "feel=DEEP    | ALL:indigo-circle(contemplating existence)"

    if command.mode == "celebration_spin":
        return "feel=PARTY   | ALL:rainbow-spin(CELEBRATION!)"

    if command.mode == "surprise_shake":
        return "feel=SHOCKED | ALL:flash-shake(whoa!)"

    if command.mode == "spiral_exploration":
        return "feel=EXPLORER| ALL:rainbow-spiral(systematic search)"

    if command.mode == "nervous_creep":
        return "feel=NERVOUS | ALL:red-flicker(nervous creep...)"

    if command.mode == "triumphant_arc":
        return "feel=TRIUMPH | ALL:gold-arc(VICTORY ARC!)"

    if command.mode == "confused_figure8":
        return "feel=CONFUSED| ALL:purple-figure8(where am I?)"

    if command.mode == "relaxed_cruise":
        return "feel=ZEN     | ALL:teal-wave(relaxed cruise~)"

    if command.mode == "alert_scan":
        return "feel=ALERT   | ALL:amber-sweep(scanning...)"

    if command.mode == "satisfied_purr":
        return "feel=CONTENT | ALL:warm-purr(satisfied purr~)"

    if command.mode == "determined_lunge":
        return "feel=DRIVEN  | ALL:orange-lunge(determined!)"

    if command.mode == "grateful_bow":
        return "feel=GRATEFUL| ALL:gold-bow(thank you~)"

    if command.mode == "victory_lap":
        return "feel=CHAMPION| ALL:rainbow-lap(VICTORY LAP!)"

    if command.mode == "curious_sniff":
        return "feel=INQUISITIVE| ALL:cyan-sniff(sniff sniff...)"

    if command.mode == "gentle_weave":
        return "feel=FLOWING | ALL:teal-weave(gentle weave~)"

    if command.mode == "happy_hop":
        return "feel=JOYFUL  | ALL:yellow-hop(happy hop!)"

    if command.mode == "boustrophedon":
        return "feel=SYSTEMATIC| ALL:green-wave(lawnmower coverage)"

    if command.mode == "zoomies":
        return "feel=ZOOMIES | ALL:rainbow-burst(ZOOM ZOOM ZOOM!)"

    if command.mode == "play_bow":
        return "feel=PLAYFUL | ALL:warm-bow(wanna play?)"

    if command.mode == "peek_a_boo":
        return "feel=PLAYFUL | ALL:cyan-flash(peek-a-boo!)"

    if command.mode == "stalk_mode":
        return "feel=PREDATOR | ALL:red-creep(stalking...)"

    if command.mode == "greeting":
        return "feel=FRIENDLY | ALL:gold-wag(hello there!)"

    if command.mode == "sleep_mode":
        return "feel=SLEEPY   | ALL:dim-blue(zzz...)"

    if command.mode == "backing_dance":
        return "feel=DANCING | ALL:magenta-dance(back it up!)"

    if command.mode == "serpentine":
        return "feel=FLOWING | ALL:rainbow-wave(serpentine~)"

    if command.mode == "explorer_pride":
        return "feel=PROUD   | ALL:gold-arc(explorer pride!)"

    if command.mode == "mapping_joy":
        return "feel=JOYFUL  | ALL:rainbow-spin(mapping joy~)"

    if command.mode == "corridor_dance":
        return "feel=DANCING | ALL:purple-wave(corridor dance!)"

    if command.mode == "open_space_celebration":
        return "feel=ECSTATIC| ALL:rainbow-burst(OPEN SPACE!)"

    if command.mode == "wall_caress":
        return "feel=GENTLE  | ALL:green-wave(wall caress~)"

    if command.mode == "discovery_spin":
        return "feel=THRILLED| ALL:rainbow-spin(discovery spin!)"

    # Front LEDs: distance emotion
    if front_distance <= 0.0 or front_distance <= cfg.danger_distance:
        front_label = "RED(danger!)"
        feeling = "scared"
    elif front_distance <= cfg.caution_distance:
        front_label = f"amber({front_distance:.0f}cm caution)"
        feeling = "alert"
    elif front_distance <= cfg.cruise_distance:
        front_label = f"green({front_distance:.0f}cm clear)"
        feeling = "content"
    else:
        front_label = f"teal({front_distance:.0f}cm open!)"
        feeling = "confident"

    # Middle LEDs: steering/direction emotion
    if command.heading < -10:
        mid_label = "orange(turning-L)"
        if feeling in ("content", "confident"):
            feeling = "curious"
    elif command.heading > 10:
        mid_label = "blue(turning-R)"
        if feeling in ("content", "confident"):
            feeling = "curious"
    else:
        mid_label = "green(straight)"

    # Rear LEDs: speed/energy
    spd = controller.current_speed
    if spd < 0.30:
        rear_label = "dim(slow)"
    elif spd < cfg.cruise_speed:
        rear_label = "teal(cruising)"
    else:
        rear_label = "bright-teal(fast!)"
        if feeling == "confident":
            feeling = "joyful!"

    return f"feel={feeling:<9} | F:{front_label:<22} M:{mid_label:<18} R:{rear_label}"


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


def apply_underlighting(tbot, controller: AutonomousCarController, command: MotionCommand, now: float):
    """Express the car's emotional state through per-LED underlighting."""
    try:
        from trilobot import (
            LIGHT_FRONT_LEFT, LIGHT_FRONT_RIGHT,
            LIGHT_MIDDLE_LEFT, LIGHT_MIDDLE_RIGHT,
            LIGHT_REAR_LEFT, LIGHT_REAR_RIGHT,
        )
    except ModuleNotFoundError:
        return

    cfg = controller.config

    def _hsv(h, s, v):
        r, g, b = hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
        return int(r * 255), int(g * 255), int(b * 255)

    if command.mode in ("follow", "follow_search"):
        # Cool blue pulse — lock-on tracking feel
        is_searching = command.mode == "follow_search"
        rate = 1.5 if is_searching else (4.0 if command.left_speed > 0.01 else 2.0)
        pulse = 0.5 + 0.5 * math.sin(now * rate)
        base_v = 0.08 if is_searching else 0.25
        col = _hsv(0.60, 1.0, base_v + 0.55 * pulse)
        tbot.fill_underlighting(*col)
        return

    if command.mode == "peek_pounce":
        pulse = 0.5 + 0.5 * math.sin(now * 10.0)
        front = (255, 140, 0)
        middle = (255, int(40 + 90 * pulse), 0)
        rear = (255, 255, int(120 + 90 * pulse))
        tbot.set_underlight(LIGHT_FRONT_LEFT, *front, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *front, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *middle, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *middle, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rear, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rear, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "brave_push":
        # Bold charge: warm white front (headlights on full), gold middle, orange rear thrust
        tbot.set_underlight(LIGHT_FRONT_LEFT,  255, 255, 180, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, 255, 255, 180, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, 255, 180,   0, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT,255, 180,   0, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT,   255, 100,   0, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT,  255, 100,   0, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "escape":
        # Alarm: blazing red front, hot amber rear, dim middle
        tbot.set_underlight(LIGHT_FRONT_LEFT,  255,  0,  0, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, 255,  0,  0, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT,  60,  0,  0, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT,  60,  0,  0, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT,   255, 90,  0, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT,  255, 90,  0, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "dead_end_recovery":
        # Panic spin: pulsing magenta
        pulse = 0.5 + 0.5 * math.sin(now * 14.0)
        col = _hsv(0.83, 1.0, 0.3 + 0.7 * pulse)
        tbot.fill_underlighting(*col)
        return

    if command.mode == "victory_dance":
        pulse = 0.5 + 0.5 * math.sin(now * 8.0)
        gold = (255, int(180 + 75 * pulse), int(40 * pulse))
        tbot.fill_underlighting(*gold)
        return

    if command.mode == "happy_wiggle":
        pulse = 0.5 + 0.5 * math.sin(now * 6.0)
        green = (int(40 * pulse), 255, int(80 + 100 * pulse))
        tbot.fill_underlighting(*green)
        return

    if command.mode == "frustrated_shimmy":
        pulse = 0.5 + 0.5 * math.sin(now * 12.0)
        angry = (int(200 + 55 * pulse), int(30 * (1.0 - pulse)), 0)
        tbot.fill_underlighting(*angry)
        return

    if command.mode == "confident_strut":
        pulse = 0.5 + 0.5 * math.sin(now * 3.0)
        purple = (int(180 + 75 * pulse), 0, int(200 + 55 * pulse))
        tbot.fill_underlighting(*purple)
        return

    if command.mode == "curious_tilt":
        pulse = 0.5 + 0.5 * math.sin(now * 5.0)
        cyan = (int(40 * pulse), int(200 + 55 * pulse), int(220 + 35 * pulse))
        tbot.fill_underlighting(*cyan)
        return

    if command.mode == "excited_bounce":
        pulse = 0.5 + 0.5 * math.sin(now * 10.0)
        yellow = (int(240 + 15 * pulse), int(220 + 35 * pulse), int(40 * (1.0 - pulse)))
        tbot.fill_underlighting(*yellow)
        return

    if command.mode == "contemplative_circle":
        pulse = 0.5 + 0.5 * math.sin(now * 2.0)
        indigo = (int(80 + 60 * pulse), int(20 * (1.0 - pulse)), int(200 + 55 * pulse))
        tbot.fill_underlighting(*indigo)
        return

    if command.mode == "celebration_spin":
        hue = (now * 4.0) % 1.0
        rgb = hsv_to_rgb(hue, 0.95, 0.8)
        tbot.fill_underlighting(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return

    if command.mode == "surprise_shake":
        pulse = 0.5 + 0.5 * math.sin(now * 25.0)
        flash = (int(255 * pulse), int(255 * (1.0 - pulse)), 0)
        tbot.fill_underlighting(*flash)
        return

    if command.mode == "spiral_exploration":
        hue = (now * 0.4) % 1.0
        rgb = hsv_to_rgb(hue, 0.85, 0.6 + 0.3 * math.sin(now * 3.0))
        tbot.fill_underlighting(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return

    if command.mode == "nervous_creep":
        pulse = 0.5 + 0.5 * math.sin(now * 8.0)
        flicker = 0.5 + 0.5 * math.sin(now * 25.0)
        front = (int(200 + 55 * flicker), int(30 * (1.0 - pulse)), 0)
        mid = (int(150 + 50 * pulse), int(20 * (1.0 - pulse)), 0)
        rear = (int(100 + 80 * flicker), int(40 * (1.0 - pulse)), 0)
        tbot.set_underlight(LIGHT_FRONT_LEFT, *front, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *front, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *mid, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mid, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rear, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rear, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "triumphant_arc":
        progress = min(1.0, (now - controller.emotional_move_start) / max(controller.config.triumphant_arc_duration, 0.01))
        hue = 0.1 + 0.08 * math.sin(progress * math.pi * 3)
        rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * math.sin(now * 6))
        arc_rgb = hsv_to_rgb((hue + 0.15) % 1.0, 0.85, 0.5 + 0.5 * progress)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(arc_rgb[0] * 255), int(arc_rgb[1] * 255), int(arc_rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(arc_rgb[0] * 255), int(arc_rgb[1] * 255), int(arc_rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rgb[0] * 200), int(rgb[1] * 200), int(rgb[2] * 200), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(arc_rgb[0] * 200), int(arc_rgb[1] * 200), int(arc_rgb[2] * 200), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "confused_figure8":
        hue = 0.7 + 0.15 * math.sin(now * 4)
        pulse = 0.5 + 0.5 * math.sin(now * 3)
        rgb = hsv_to_rgb(hue, 0.5, 0.4 + 0.3 * pulse)
        alt_rgb = hsv_to_rgb((hue + 0.2) % 1.0, 0.4, 0.3 + 0.4 * pulse)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(alt_rgb[0] * 255), int(alt_rgb[1] * 255), int(alt_rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(alt_rgb[0] * 255), int(alt_rgb[1] * 255), int(alt_rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rgb[0] * 200), int(rgb[1] * 200), int(rgb[2] * 200), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(alt_rgb[0] * 200), int(alt_rgb[1] * 200), int(alt_rgb[2] * 200), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "relaxed_cruise":
        hue = 0.35 + 0.1 * math.sin(now * 1.5)
        pulse = 0.5 + 0.5 * math.sin(now * 2)
        rgb = hsv_to_rgb(hue, 0.6, 0.5 + 0.3 * pulse)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(rgb[0] * 220), int(rgb[1] * 220), int(rgb[2] * 220), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(rgb[0] * 220), int(rgb[1] * 220), int(rgb[2] * 220), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rgb[0] * 180), int(rgb[1] * 180), int(rgb[2] * 180), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rgb[0] * 180), int(rgb[1] * 180), int(rgb[2] * 180), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "alert_scan":
        pulse = 0.5 + 0.5 * math.sin(now * 12.0)
        sweep = 0.5 + 0.5 * math.sin(now * 4.0)
        front = (int(200 + 55 * pulse), int(160 + 60 * sweep), 0)
        mid = (int(180 + 40 * pulse), int(140 + 40 * sweep), int(30 * pulse))
        rear = (int(150 + 50 * pulse), int(120 + 50 * sweep), int(20 * pulse))
        tbot.set_underlight(LIGHT_FRONT_LEFT, *front, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *front, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *mid, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mid, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rear, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rear, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "satisfied_purr":
        purr = 0.5 + 0.5 * math.sin(now * 6.0)
        warm = (int(220 + 35 * purr), int(140 + 60 * purr), int(40 * (1.0 - purr)))
        tbot.set_underlight(LIGHT_FRONT_LEFT, *warm, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *warm, show=False)
        mid_warm = (int(200 + 30 * purr), int(120 + 50 * purr), int(30 * (1.0 - purr)))
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *mid_warm, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mid_warm, show=False)
        rear_warm = (int(180 + 40 * purr), int(100 + 40 * purr), int(20 * (1.0 - purr)))
        tbot.set_underlight(LIGHT_REAR_LEFT, *rear_warm, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rear_warm, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "determined_lunge":
        progress = min(1.0, (now - controller.emotional_move_start) / max(controller.config.determined_lunge_duration, 0.01))
        intensity = 0.3 + 0.7 * progress
        rgb = hsv_to_rgb(0.08, 0.9, intensity)
        tbot.fill_underlighting(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return

    if command.mode == "grateful_bow":
        bow = math.sin((now - controller.emotional_move_start) / max(controller.config.grateful_bow_duration, 0.01) * math.pi)
        gold = (int(200 + 55 * bow), int(160 + 70 * bow), int(40 * bow))
        tbot.fill_underlighting(*gold)
        return

    if command.mode == "victory_lap":
        hue = (now * 0.6) % 1.0
        rgb = hsv_to_rgb(hue, 0.95, 0.7 + 0.3 * math.sin(now * 6))
        tbot.fill_underlighting(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return

    if command.mode == "curious_sniff":
        sniff = 0.5 + 0.5 * math.sin(now * 5.0)
        cyan = (int(40 * sniff), int(180 + 75 * sniff), int(200 + 55 * sniff))
        tbot.fill_underlighting(*cyan)
        return

    if command.mode == "gentle_weave":
        weave = 0.5 + 0.5 * math.sin(now * 3.0)
        teal = (int(40 * (1.0 - weave)), int(200 + 55 * weave), int(180 + 40 * weave))
        tbot.fill_underlighting(*teal)
        return

    if command.mode == "happy_hop":
        hop = 0.5 + 0.5 * math.sin(now * 10.0)
        yellow = (int(240 + 15 * hop), int(220 + 35 * hop), int(30 * (1.0 - hop)))
        tbot.fill_underlighting(*yellow)
        return

    if command.mode == "boustrophedon":
        wave = 0.5 + 0.5 * math.sin(now * 2.0)
        progress = min(1.0, (now - controller.boustrophedon_start_time) / max(controller.config.boustrophedon_duration_s, 0.01))
        hue = 0.25 + 0.15 * math.sin(progress * math.pi * 3)
        front = _hsv(hue, 0.85, 0.5 + 0.4 * wave)
        mid = _hsv((hue + 0.05) % 1.0, 0.8, 0.4 + 0.3 * wave)
        rear = _hsv((hue + 0.1) % 1.0, 0.75, 0.3 + 0.3 * wave)
        tbot.set_underlight(LIGHT_FRONT_LEFT, *front, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *front, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *mid, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mid, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rear, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rear, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "zoomies":
        hue = (now * 5.0) % 1.0
        burst = 0.5 + 0.5 * math.sin(now * 15.0)
        fl = _hsv(hue, 0.95, 0.7 + 0.3 * burst)
        fr = _hsv((hue + 0.15) % 1.0, 0.95, 0.7 + 0.3 * burst)
        ml = _hsv((hue + 0.3) % 1.0, 0.9, 0.6 + 0.4 * burst)
        mr = _hsv((hue + 0.45) % 1.0, 0.9, 0.6 + 0.4 * burst)
        rl = _hsv((hue + 0.6) % 1.0, 0.85, 0.5 + 0.5 * burst)
        rr = _hsv((hue + 0.75) % 1.0, 0.85, 0.5 + 0.5 * burst)
        tbot.set_underlight(LIGHT_FRONT_LEFT, *fl, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *fr, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *ml, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mr, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rl, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rr, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "play_bow":
        bow = math.sin((now - controller.emotional_move_start) / max(controller.config.play_bow_duration, 0.01) * math.pi)
        warm = (int(200 + 55 * bow), int(140 + 80 * bow), int(40 * (1.0 - bow)))
        tbot.fill_underlighting(*warm)
        return

    if command.mode == "peek_a_boo":
        peek = 0.5 + 0.5 * math.sin(now * 6.0)
        if peek > 0.5:
            cyan = (int(40 * peek), int(200 + 55 * peek), int(220 + 35 * peek))
            tbot.fill_underlighting(*cyan)
        else:
            tbot.fill_underlighting(20, 20, 40)
        return

    if command.mode == "stalk_mode":
        progress = min(1.0, (now - controller.emotional_move_start) / max(controller.config.stalk_mode_duration, 0.01))
        intensity = 0.2 + 0.6 * progress
        flicker = 0.5 + 0.5 * math.sin(now * 8.0)
        front = (int(180 + 75 * flicker), int(30 * intensity), 0)
        mid = (int(140 + 60 * flicker), int(20 * intensity), 0)
        rear = (int(100 + 50 * flicker), int(15 * intensity), 0)
        tbot.set_underlight(LIGHT_FRONT_LEFT, *front, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *front, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *mid, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mid, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rear, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rear, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "greeting":
        wag = 0.5 + 0.5 * math.sin(now * 8.0)
        hue = 0.12 + 0.08 * math.sin(now * 3.0)
        rgb = hsv_to_rgb(hue, 0.85, 0.5 + 0.4 * wag)
        tbot.fill_underlighting(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return

    if command.mode == "sleep_mode":
        breathe = 0.5 + 0.5 * math.sin(now * 1.5)
        blue = (int(20 * breathe), int(30 * breathe), int(80 + 60 * breathe))
        tbot.fill_underlighting(*blue)
        return

    if command.mode == "backing_dance":
        dance = 0.5 + 0.5 * math.sin(now * 10.0)
        hue = 0.8 + 0.1 * math.sin(now * 5.0)
        fl = _hsv(hue, 0.7, 0.4 + 0.4 * dance)
        fr = _hsv((hue + 0.2) % 1.0, 0.7, 0.4 + 0.4 * dance)
        ml = _hsv((hue + 0.1) % 1.0, 0.65, 0.3 + 0.3 * dance)
        mr = _hsv((hue + 0.3) % 1.0, 0.65, 0.3 + 0.3 * dance)
        rl = _hsv((hue + 0.15) % 1.0, 0.6, 0.2 + 0.2 * dance)
        rr = _hsv((hue + 0.35) % 1.0, 0.6, 0.2 + 0.2 * dance)
        tbot.set_underlight(LIGHT_FRONT_LEFT, *fl, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *fr, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *ml, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mr, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rl, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rr, show=False)
        tbot.show_underlighting()
        return

    if command.mode == "serpentine":
        hue = (now * 0.8) % 1.0
        wave = 0.5 + 0.5 * math.sin(now * 3.0)
        fl = _hsv(hue, 0.85, 0.6 + 0.3 * wave)
        fr = _hsv((hue + 0.12) % 1.0, 0.85, 0.6 + 0.3 * wave)
        ml = _hsv((hue + 0.24) % 1.0, 0.8, 0.5 + 0.3 * wave)
        mr = _hsv((hue + 0.36) % 1.0, 0.8, 0.5 + 0.3 * wave)
        rl = _hsv((hue + 0.48) % 1.0, 0.75, 0.4 + 0.3 * wave)
        rr = _hsv((hue + 0.6) % 1.0, 0.75, 0.4 + 0.3 * wave)
        tbot.set_underlight(LIGHT_FRONT_LEFT, *fl, show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, *fr, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, *ml, show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *mr, show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, *rl, show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, *rr, show=False)
        tbot.show_underlighting()
        return

    # Gen23: new emotional move underlighting
    if command.mode == "explorer_pride":
        progress = min(1.0, (now - controller.emotional_move_start) / max(controller.config.explorer_pride_duration, 0.01))
        hue = 0.12 + 0.08 * math.sin(progress * math.pi)
        rgb = hsv_to_rgb(hue, 0.9, 0.6 + 0.4 * math.sin(now * 4))
        fl = _hsv(hue, 0.9, 0.7 + 0.3 * progress)
        fr = _hsv((hue + 0.05) % 1.0, 0.9, 0.7 + 0.3 * progress)
        ml = _hsv((hue + 0.1) % 1.0, 0.85, 0.5 + 0.3 * progress)
        mr = _hsv((hue + 0.15) % 1.0, 0.85, 0.5 + 0.3 * progress)
        rl = _hsv((hue + 0.2) % 1.0, 0.8, 0.4 + 0.2 * progress)
        rr = _hsv((hue + 0.25) % 1.0, 0.8, 0.4 + 0.2 * progress)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(fl[0] * 255), int(fl[1] * 255), int(fl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(fr[0] * 255), int(fr[1] * 255), int(fr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(ml[0] * 255), int(ml[1] * 255), int(ml[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(mr[0] * 255), int(mr[1] * 255), int(mr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rl[0] * 255), int(rl[1] * 255), int(rl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rr[0] * 255), int(rr[1] * 255), int(rr[2] * 255), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "mapping_joy":
        hue = (now * 0.5) % 1.0
        wave = 0.5 + 0.5 * math.sin(now * 4.0)
        fl = _hsv(hue, 0.85, 0.6 + 0.3 * wave)
        fr = _hsv((hue + 0.17) % 1.0, 0.85, 0.6 + 0.3 * wave)
        ml = _hsv((hue + 0.33) % 1.0, 0.8, 0.5 + 0.3 * wave)
        mr = _hsv((hue + 0.5) % 1.0, 0.8, 0.5 + 0.3 * wave)
        rl = _hsv((hue + 0.67) % 1.0, 0.75, 0.4 + 0.2 * wave)
        rr = _hsv((hue + 0.83) % 1.0, 0.75, 0.4 + 0.2 * wave)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(fl[0] * 255), int(fl[1] * 255), int(fl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(fr[0] * 255), int(fr[1] * 255), int(fr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(ml[0] * 255), int(ml[1] * 255), int(ml[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(mr[0] * 255), int(mr[1] * 255), int(mr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rl[0] * 255), int(rl[1] * 255), int(rl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rr[0] * 255), int(rr[1] * 255), int(rr[2] * 255), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "corridor_dance":
        hue = 0.3 + 0.15 * math.sin(now * 5.0)
        wave = 0.5 + 0.5 * math.sin(now * 6.0)
        fl = _hsv(hue, 0.85, 0.6 + 0.3 * wave)
        fr = _hsv((hue + 0.1) % 1.0, 0.85, 0.6 + 0.3 * wave)
        ml = _hsv((hue + 0.2) % 1.0, 0.8, 0.5 + 0.3 * wave)
        mr = _hsv((hue + 0.3) % 1.0, 0.8, 0.5 + 0.3 * wave)
        rl = _hsv((hue + 0.4) % 1.0, 0.75, 0.4 + 0.2 * wave)
        rr = _hsv((hue + 0.5) % 1.0, 0.75, 0.4 + 0.2 * wave)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(fl[0] * 255), int(fl[1] * 255), int(fl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(fr[0] * 255), int(fr[1] * 255), int(fr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(ml[0] * 255), int(ml[1] * 255), int(ml[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(mr[0] * 255), int(mr[1] * 255), int(mr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rl[0] * 255), int(rl[1] * 255), int(rl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rr[0] * 255), int(rr[1] * 255), int(rr[2] * 255), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "open_space_celebration":
        hue = (now * 3.0) % 1.0
        burst = 0.5 + 0.5 * math.sin(now * 12.0)
        fl = _hsv(hue, 0.95, 0.7 + 0.3 * burst)
        fr = _hsv((hue + 0.17) % 1.0, 0.95, 0.7 + 0.3 * burst)
        ml = _hsv((hue + 0.33) % 1.0, 0.9, 0.6 + 0.4 * burst)
        mr = _hsv((hue + 0.5) % 1.0, 0.9, 0.6 + 0.4 * burst)
        rl = _hsv((hue + 0.67) % 1.0, 0.85, 0.5 + 0.5 * burst)
        rr = _hsv((hue + 0.83) % 1.0, 0.85, 0.5 + 0.5 * burst)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(fl[0] * 255), int(fl[1] * 255), int(fl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(fr[0] * 255), int(fr[1] * 255), int(fr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(ml[0] * 255), int(ml[1] * 255), int(ml[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(mr[0] * 255), int(mr[1] * 255), int(mr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rl[0] * 255), int(rl[1] * 255), int(rl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rr[0] * 255), int(rr[1] * 255), int(rr[2] * 255), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "wall_caress":
        hue = 0.25 + 0.08 * math.sin(now * 3.0)
        wave = 0.5 + 0.5 * math.sin(now * 2.5)
        fl = _hsv(hue, 0.7, 0.5 + 0.3 * wave)
        fr = _hsv(hue, 0.7, 0.5 + 0.3 * wave)
        ml = _hsv((hue + 0.05) % 1.0, 0.65, 0.4 + 0.2 * wave)
        mr = _hsv((hue + 0.05) % 1.0, 0.65, 0.4 + 0.2 * wave)
        rl = _hsv((hue + 0.1) % 1.0, 0.6, 0.3 + 0.2 * wave)
        rr = _hsv((hue + 0.1) % 1.0, 0.6, 0.3 + 0.2 * wave)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(fl[0] * 255), int(fl[1] * 255), int(fl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(fr[0] * 255), int(fr[1] * 255), int(fr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(ml[0] * 255), int(ml[1] * 255), int(ml[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(mr[0] * 255), int(mr[1] * 255), int(mr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rl[0] * 255), int(rl[1] * 255), int(rl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rr[0] * 255), int(rr[1] * 255), int(rr[2] * 255), show=False)
        tbot.show_underlighting()
        return

    if command.mode == "discovery_spin":
        hue = (now * 4.0) % 1.0
        spin = 0.5 + 0.5 * math.sin(now * 10.0)
        fl = _hsv(hue, 0.9, 0.7 + 0.3 * spin)
        fr = _hsv((hue + 0.25) % 1.0, 0.9, 0.7 + 0.3 * spin)
        ml = _hsv((hue + 0.5) % 1.0, 0.85, 0.6 + 0.3 * spin)
        mr = _hsv((hue + 0.75) % 1.0, 0.85, 0.6 + 0.3 * spin)
        rl = _hsv(hue, 0.8, 0.5 + 0.3 * spin)
        rr = _hsv((hue + 0.5) % 1.0, 0.8, 0.5 + 0.3 * spin)
        tbot.set_underlight(LIGHT_FRONT_LEFT, int(fl[0] * 255), int(fl[1] * 255), int(fl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_FRONT_RIGHT, int(fr[0] * 255), int(fr[1] * 255), int(fr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_LEFT, int(ml[0] * 255), int(ml[1] * 255), int(ml[2] * 255), show=False)
        tbot.set_underlight(LIGHT_MIDDLE_RIGHT, int(mr[0] * 255), int(mr[1] * 255), int(mr[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_LEFT, int(rl[0] * 255), int(rl[1] * 255), int(rl[2] * 255), show=False)
        tbot.set_underlight(LIGHT_REAR_RIGHT, int(rr[0] * 255), int(rr[1] * 255), int(rr[2] * 255), show=False)
        tbot.show_underlighting()
        return

    # --- drive mode ---
    front_dist = controller.sanitize_distance(controller.last_scan.get(0, cfg.cruise_distance))
    speed = controller.current_speed
    speed_ratio = min(1.0, speed / max(cfg.open_space_speed, 0.01))
    max_angle = max(abs(a) for a in cfg.scan_angles) or 1
    steer = command.heading / max_angle  # −1=full left … +1=full right

    # Front LEDs: distance emotion — red danger → amber caution → green clear → bright teal open
    if front_dist <= 0.0 or front_dist <= cfg.danger_distance:
        front = (255, 0, 0)
    elif front_dist <= cfg.caution_distance:
        t = (front_dist - cfg.danger_distance) / max(cfg.caution_distance - cfg.danger_distance, 1.0)
        front = (255, int(110 * t), 0)
    elif front_dist <= cfg.cruise_distance:
        t = (front_dist - cfg.caution_distance) / max(cfg.cruise_distance - cfg.caution_distance, 1.0)
        front = (int(255 * (1.0 - t)), 255, 0)
    else:
        extra = min(1.0, (front_dist - cfg.cruise_distance) / 40.0)
        front = (0, 255, int(180 * extra))

    # Middle LEDs: steering emotion — orange=turning left, blue=turning right, green=straight
    left_turn  = max(0.0, -steer)
    right_turn = max(0.0,  steer)
    mid_l = _hsv(
        0.08 * left_turn + 0.33 * (1.0 - left_turn),   # orange → green
        0.9,
        0.2 + 0.55 * speed_ratio + 0.25 * left_turn,
    )
    mid_r = _hsv(
        0.60 * right_turn + 0.33 * (1.0 - right_turn), # blue → green
        0.9,
        0.2 + 0.55 * speed_ratio + 0.25 * right_turn,
    )

    # Rear LEDs: speed/energy — teal thruster glow, breathing pulse with speed
    pulse = 0.5 + 0.5 * math.sin(now * (2.5 + speed_ratio * 5.0))
    rear_v = 0.05 + 0.55 * speed_ratio + 0.25 * pulse * speed_ratio
    rear = _hsv(0.50 + 0.06 * speed_ratio, 0.95, rear_v)

    # Gen18: Mood-based hue overlay
    mood_hue = controller.get_mood_hue_offset(now)
    mood_influence = controller.config.mood_personality_color_influence
    if controller.current_mood == "celebration":
        front_r, front_g, front_b = front
        celeb_h = mood_hue
        celeb_rgb = hsv_to_rgb(celeb_h, 0.8, 0.7 + 0.3 * pulse)
        front = (
            min(255, int(front_r * (1.0 - mood_influence) + celeb_rgb[0] * 255 * mood_influence)),
            min(255, int(front_g * (1.0 - mood_influence) + celeb_rgb[1] * 255 * mood_influence)),
            min(255, int(front_b * (1.0 - mood_influence) + celeb_rgb[2] * 255 * mood_influence)),
        )
    elif controller.current_mood == "stressed":
        front_r, front_g, front_b = front
        stress_wash = controller.config.mood_stress_wash_intensity
        front = (
            min(255, int(front_r + (255 - front_r) * stress_wash)),
            int(front_g * (1.0 - stress_wash)),
            int(front_b * (1.0 - stress_wash)),
        )
    elif controller.current_mood == "exploring":
        explore_h = mood_hue
        explore_rgb = hsv_to_rgb(explore_h, 0.6, 0.3)
        front_r, front_g, front_b = front
        front = (
            min(255, int(front_r * (1.0 - mood_influence) + explore_rgb[0] * 255 * mood_influence)),
            min(255, int(front_g * (1.0 - mood_influence) + explore_rgb[1] * 255 * mood_influence)),
            min(255, int(front_b * (1.0 - mood_influence) + explore_rgb[2] * 255 * mood_influence)),
        )

    # Gen20: Per-LED emotional expressions with rainbow wave effects
    blend = controller.config.mood_led_gradient_blend
    wave_rate = controller.config.mood_led_wave_rate
    wave_phase = now * wave_rate

    # Celebration: rainbow wave across all LEDs
    if controller.current_mood == "celebration":
        rainbow_speed = controller.config.mood_celebration_rainbow_speed
        fl_h = (wave_phase * 0.3) % 1.0
        fr_h = (wave_phase * 0.3 + 0.17) % 1.0
        ml_h = (wave_phase * 0.3 + 0.33) % 1.0
        mr_h = (wave_phase * 0.3 + 0.5) % 1.0
        rl_h = (wave_phase * 0.3 + 0.67) % 1.0
        rr_h = (wave_phase * 0.3 + 0.83) % 1.0
        celeb_v = 0.6 + 0.4 * math.sin(now * 4)
        front = _hsv(fl_h, 0.9, celeb_v)
        mid_l = _hsv(ml_h, 0.85, celeb_v * 0.9)
        mid_r = _hsv(mr_h, 0.85, celeb_v * 0.9)
        rear = _hsv(rl_h, 0.8, celeb_v * 0.7)

    # Confident: purple glow pulse
    elif controller.current_mood == "confident":
        glow_rate = controller.config.mood_confident_glow_rate
        glow = 0.5 + 0.5 * math.sin(now * glow_rate)
        purple_base = 0.78
        front_r, front_g, front_b = front
        purple = _hsv(purple_base, 0.8, 0.4 + 0.4 * glow)
        front = (
            min(255, int(front_r * (1.0 - blend) + purple[0] * blend)),
            min(255, int(front_g * (1.0 - blend) + purple[1] * blend)),
            min(255, int(front_b * (1.0 - blend) + purple[2] * blend)),
        )
        mid_l = _hsv(purple_base + 0.03, 0.7, 0.3 + 0.3 * glow)
        mid_r = _hsv(purple_base - 0.03, 0.7, 0.3 + 0.3 * glow)
        rear = _hsv(purple_base, 0.6, 0.2 + 0.2 * glow)

    # Exploring: hue rotation wave
    elif controller.current_mood == "exploring":
        hue_speed = controller.config.mood_exploring_hue_speed
        explore_hue = (now * hue_speed) % 1.0
        explore_rgb_wave = hsv_to_rgb(explore_hue, 0.7, 0.4)
        front_r, front_g, front_b = front
        front = (
            min(255, int(front_r * (1.0 - blend) + explore_rgb_wave[0] * blend)),
            min(255, int(front_g * (1.0 - blend) + explore_rgb_wave[1] * blend)),
            min(255, int(front_b * (1.0 - blend) + explore_rgb_wave[2] * blend)),
        )
        mid_l = _hsv((explore_hue + 0.15) % 1.0, 0.6, 0.3)
        mid_r = _hsv((explore_hue + 0.3) % 1.0, 0.6, 0.3)
        rear = _hsv((explore_hue + 0.45) % 1.0, 0.5, 0.2)

    # Stressed: red flicker
    elif controller.current_mood == "stressed":
        flicker_rate = controller.config.mood_stressed_flicker_rate
        flicker = 0.5 + 0.5 * math.sin(now * flicker_rate + math.sin(now * 17) * 3)
        front_r, front_g, front_b = front
        stress_pulse = int(flicker * 80)
        front = (min(255, front_r + stress_pulse), max(0, front_g - stress_pulse // 2), 0)
        mid_l = (int(120 + 60 * flicker), int(20 * (1.0 - flicker)), 0)
        mid_r = (int(120 + 60 * flicker), int(20 * (1.0 - flicker)), 0)
        rear = (int(80 + 40 * flicker), int(10 * (1.0 - flicker)), 0)

    # Stressed overlay: wash everything toward red as stress rises
    def _stress(color):
        r, g, b = color
        s = controller.stress
        r = min(255, r + int(s * 120))
        g = int(g * (1.0 - s * 0.5))
        b = int(b * (1.0 - s * 0.6))
        return r, g, b

    tbot.set_underlight(LIGHT_FRONT_LEFT,   *_stress(front), show=False)
    tbot.set_underlight(LIGHT_FRONT_RIGHT,  *_stress(front), show=False)
    tbot.set_underlight(LIGHT_MIDDLE_LEFT,  *_stress(mid_l), show=False)
    tbot.set_underlight(LIGHT_MIDDLE_RIGHT, *_stress(mid_r), show=False)
    tbot.set_underlight(LIGHT_REAR_LEFT,    *_stress(rear),  show=False)
    tbot.set_underlight(LIGHT_REAR_RIGHT,   *_stress(rear),  show=False)
    tbot.show_underlighting()


def apply_command(tbot, controller: AutonomousCarController, command: MotionCommand, now: float | None = None):
    if now is None:
        now = time.monotonic()
    apply_underlighting(tbot, controller, command, now)

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

    if command.mode in ("follow", "follow_search", "peek_pounce"):
        tbot.set_motor_speeds(command.left_speed, command.right_speed)
        return

    if command.mode == "brave_push":
        tbot.set_motor_speeds(command.left_speed, command.right_speed)
        time.sleep(controller.config.push_duration_s)
        tbot.stop()
        controller.last_scan_time = -math.inf  # force immediate rescan to see if it moved
        return

    if command.mode == "dead_end_recovery":
        controller.note_turn(command.heading)
        tbot.set_motor_speeds(command.left_speed, command.right_speed)
        time.sleep(controller.config.dead_end_recovery_turn_s)
        tbot.stop()
        return

    if command.mode in ("victory_dance", "happy_wiggle", "frustrated_shimmy", "confident_strut",
                        "curious_tilt", "excited_bounce", "contemplative_circle", "celebration_spin",
                        "surprise_shake", "spiral_exploration", "nervous_creep", "triumphant_arc",
                        "confused_figure8", "relaxed_cruise", "alert_scan",
                        "satisfied_purr", "determined_lunge", "grateful_bow", "victory_lap",
                        "curious_sniff", "gentle_weave", "happy_hop",
                        "explorer_pride", "mapping_joy", "corridor_dance",
                        "open_space_celebration", "wall_caress", "discovery_spin"):
        tbot.set_motor_speeds(command.left_speed, command.right_speed)
        time.sleep(0.08)
        tbot.stop()
        return

    if abs(command.heading) >= 30:
        controller.note_turn(command.heading)

    tbot.set_motor_speeds(command.left_speed, command.right_speed)


def startup_spin_scan(tbot, controller: AutonomousCarController):
    """Spin 360° in four 90° steps, scanning at each quadrant.
    Ends facing the most open direction found."""
    cfg = controller.config
    turn_s = 0.46  # estimated time for ~90° turn
    print("[SPIN] Scanning room at startup...")
    quad_front: list = []

    for i in range(4):
        perform_scan(tbot, controller)
        quad_front.append(controller.sanitize_distance(controller.last_scan.get(0, 0.0)))
        _print_scan(controller.last_scan)
        print(f"[HEAT] {controller.format_heatmap()}  (danger memory, █=avoid)")
        tbot.turn_left(cfg.escape_turn_speed)
        time.sleep(turn_s)
        tbot.stop()
        time.sleep(0.08)

    # Turn to face the most open quadrant
    best = quad_front.index(max(quad_front))
    turns_to_best = (1 + best) % 4   # we ended 270° left of start; 1 more turn = 0°
    print(f"[SPIN] Best heading: quadrant {best} ({best*90}°), front={quad_front[best]:.0f}cm")
    for _ in range(turns_to_best):
        tbot.turn_left(cfg.escape_turn_speed)
        time.sleep(turn_s)
        tbot.stop()
        time.sleep(0.08)

    perform_scan(tbot, controller)
    _print_scan(controller.last_scan)
    print(f"[HEAT] {controller.format_heatmap()}  (danger memory, █=avoid)")
    print("[SPIN] Ready — starting navigation.\n")


def main():
    from trilobot import BUTTON_A, BUTTON_X, BUTTON_Y, Trilobot

    print("Trilobot Example: Autonomous Car")
    print("  A = toggle follow mode   X = stop   Y = distance_lights\n")

    tbot = Trilobot()
    controller = AutonomousCarController(AutonomousCarConfig())
    on_inline_line = False
    last_print_time = 0.0
    print_interval = 0.3
    launch_distance_lights = False
    follow_mode = False
    btn_a_prev = False

    try:
        tbot.initialise_servo()
        tbot.set_servo_angle(0)
        time.sleep(0.25)
        startup_spin_scan(tbot, controller)

        while not tbot.read_button(BUTTON_X):
            # Button A: toggle follow mode (debounced)
            btn_a_now = tbot.read_button(BUTTON_A)
            if btn_a_now and not btn_a_prev:
                follow_mode = not follow_mode
                if on_inline_line:
                    print()
                    on_inline_line = False
                label = "FOLLOW mode ON  (A again to return)" if follow_mode else "AUTONOMOUS mode"
                print(f"[A] {label}")
            btn_a_prev = btn_a_now

            if tbot.read_button(BUTTON_Y):
                launch_distance_lights = True
                break

            now = time.monotonic()
            front_distance = tbot.read_distance(
                timeout=controller.config.front_timeout_ms,
                samples=controller.config.front_samples,
            )

            if follow_mode:
                command = controller.plan_follow(front_distance)
                emotion = _describe_lights(command, controller, front_distance)
                if now - last_print_time >= print_interval:
                    fd_str = f"{front_distance:.0f}cm" if front_distance > 0 else " ---"
                    err = front_distance - controller.config.follow_target_distance if front_distance > 0 else 0.0
                    if command.left_speed > 0.01:
                        action = "chasing →"
                    elif command.left_speed < -0.01:
                        action = "← backing"
                    else:
                        action = "holding  "
                    stress_ch = ' ░▒▓█'[min(4, int(controller.stress * 5))]
                    print(f"\r[FOLLOW] front={fd_str}  err={err:+.0f}cm  {action}  stress={stress_ch}  | {emotion}   ", end='', flush=True)
                    on_inline_line = True
                    last_print_time = now
                apply_command(tbot, controller, command, now)
            else:
                if controller.last_scan:
                    controller.last_scan[0] = controller.sanitize_distance(front_distance)

                if controller.should_scan(front_distance, now):
                    perform_scan(tbot, controller)
                    front_distance = controller.last_scan.get(0, front_distance)
                    if on_inline_line:
                        print()
                        on_inline_line = False
                    _print_scan(controller.last_scan)
                    print(f"[HEAT] {controller.format_heatmap()}  (danger memory, █=avoid)")

                command = controller.plan(front_distance, now)
                emotion = _describe_lights(command, controller, front_distance)
                stress_ch = ' ░▒▓█'[min(4, int(controller.stress * 5))]
                quip = controller.get_quip(command.mode, front_distance, command.heading, now)

                if command.mode == "brave_push":
                    if on_inline_line:
                        print()
                        on_inline_line = False
                    print(f"[PUSH!]    front={front_distance:.0f}cm  → {quip if quip else 'CHARGE!'}  stress={stress_ch}  | {emotion}")
                elif command.mode == "peek_pounce":
                    if on_inline_line:
                        print()
                        on_inline_line = False
                    side = "L" if command.heading < 0 else "R"
                    print(f"[POUNCE]   front={front_distance:.0f}cm  →{side}  {quip if quip else 'pounce!'}  stress={stress_ch}  | {emotion}")
                elif command.mode == "escape":
                    if on_inline_line:
                        print()
                        on_inline_line = False
                    esc_lvl = controller.get_escape_escalation_level(now)
                    side = "L" if command.heading < 0 else "R"
                    print(f"[ESCAPE-{esc_lvl}] front={front_distance:.0f}cm  →{side}  {quip if quip else 'evasive!'}  stress={stress_ch}  | {emotion}")
                elif command.mode == "dead_end_recovery":
                    if on_inline_line:
                        print()
                        on_inline_line = False
                    side = "L" if command.heading < 0 else "R"
                    print(f"[DEAD END] front={front_distance:.0f}cm  →{side}  {quip if quip else 'this wall again?'}  stress={stress_ch}  | {emotion}")
                elif command.mode in ("victory_dance", "happy_wiggle", "frustrated_shimmy", "confident_strut",
                                       "curious_tilt", "excited_bounce", "contemplative_circle", "celebration_spin",
                                       "surprise_shake", "spiral_exploration", "nervous_creep", "triumphant_arc",
                                       "confused_figure8", "relaxed_cruise", "alert_scan",
                                       "satisfied_purr", "determined_lunge", "grateful_bow", "victory_lap",
                                       "curious_sniff", "gentle_weave", "happy_hop",
                                        "boustrophedon", "zoomies", "play_bow", "peek_a_boo", "stalk_mode",
                                        "greeting", "sleep_mode", "backing_dance", "serpentine",
                                        "explorer_pride", "mapping_joy", "corridor_dance",
                                        "open_space_celebration", "wall_caress", "discovery_spin"):
                    if on_inline_line:
                        print()
                        on_inline_line = False
                    move_labels = {
                        "victory_dance": "VICTORY DANCE",
                        "happy_wiggle": "HAPPY WIGGLE",
                        "frustrated_shimmy": "FRUSTRATED SHIMMY",
                        "confident_strut": "CONFIDENT STRUT",
                        "curious_tilt": "CURIOUS TILT",
                        "excited_bounce": "EXCITED BOUNCE",
                        "contemplative_circle": "CONTEMPLATIVE CIRCLE",
                        "celebration_spin": "CELEBRATION SPIN",
                        "surprise_shake": "SURPRISE SHAKE",
                        "spiral_exploration": "SPIRAL EXPLORATION",
                        "nervous_creep": "NERVOUS CREEP",
                        "triumphant_arc": "TRIUMPHANT ARC",
                        "confused_figure8": "CONFUSED FIGURE-8",
                        "relaxed_cruise": "RELAXED CRUISE",
                        "alert_scan": "ALERT SCAN",
                        "satisfied_purr": "SATISFIED PURR",
                        "determined_lunge": "DETERMINED LUNGE",
                        "grateful_bow": "GRATEFUL BOW",
                        "victory_lap": "VICTORY LAP",
                        "curious_sniff": "CURIOUS SNIFF",
                        "gentle_weave": "GENTLE WEAVE",
                        "happy_hop": "HAPPY HOP",
                        "boustrophedon": "BOUSTROPHEDON",
                        "zoomies": "ZOOMIES",
                        "play_bow": "PLAY BOW",
                        "peek_a_boo": "PEEK-A-BOO",
                        "stalk_mode": "STALK MODE",
                        "greeting": "GREETING",
                        "sleep_mode": "SLEEP MODE",
                        "backing_dance": "BACKING DANCE",
                        "serpentine": "SERPENTINE",
                        "explorer_pride": "EXPLORER PRIDE",
                        "mapping_joy": "MAPPING JOY",
                        "corridor_dance": "CORRIDOR DANCE",
                        "open_space_celebration": "OPEN SPACE CELEBRATION",
                        "wall_caress": "WALL CARESS",
                        "discovery_spin": "DISCOVERY SPIN",
                    }
                    label = move_labels.get(command.mode, command.mode)
                    print(f"[{label}] front={front_distance:.0f}cm  mood={controller.current_mood}  | {emotion}")
                else:
                    if now - last_print_time >= print_interval:
                        arrow = "←" if command.heading < -10 else ("→" if command.heading > 10 else "↑")
                        terrain_tag = f" [{controller.terrain_class}]" if controller.terrain_class != "unknown" else ""
                        corner_tag = " [corner]" if controller.corner_detected else ""
                        momentum_tag = f" mom={controller.heading_momentum:+.2f}" if abs(controller.heading_momentum) > 0.2 else ""
                        personality = controller.get_personality_label()
                        streak = f" streak={controller.success_streak}" if controller.success_streak >= 3 else ""
                        quip_tag = f"  «{quip}»" if quip else ""
                        closing_tag = f"  [closing]" if controller.last_side_closing > 0.5 else ""
                        exit_tag = f"  [exit→{controller.corridor_exit_heading:+d}]" if controller.corridor_exit_detected else ""
                        recovery_tag = f"  [recovery:{controller.recovery_stage}]" if controller.recovery_stage > 0 else ""
                        dead_end_pred_tag = "  [dead-end predicted]" if controller.dead_end_predicted else ""
                        shape_tag = f"  [{controller.obstacle_shape}]" if controller.obstacle_shape != "unknown" else ""
                        achievement_tag = f"  [🏆{controller.achievement_count}]" if controller.achievement_count > 0 else ""
                        junction_tag = f"  [{controller.junction_type}J]" if controller.junction_detected else ""
                        strategy_tag = f"  [{controller.current_exploration_strategy}]" if controller.current_exploration_strategy != "balanced" else ""
                        patience_tag = "  [patience]" if controller.tactical_patience_active else ""
                        nearmiss_tag = f"  [near-miss×{len(controller.near_miss_history)}]" if len(controller.near_miss_history) >= 3 else ""
                        loop_tag = f"  [loop×{controller.loop_count}]" if controller.loop_count > 0 and now - controller.last_loop_detection_time < 15.0 else ""
                        wf_tag = "  [wall-follow]" if controller.wall_following_escape_active else ""
                        moving_tag = f"  [moving@{','.join(str(a) for a in sorted(controller.moving_obstacles))}]" if controller.moving_obstacles else ""
                        hysteresis_tag = f"  [hysteresis×{controller.heading_hysteresis_count}]" if controller.heading_hysteresis_count > 0 else ""
                        side_corridor_tag = f"  [side→{controller.side_corridor_heading:+d}]" if controller.side_corridor_detected else ""
                        cluster_tag = f"  [cluster×{controller.obstacle_cluster_tightness:.1f}]" if controller.obstacle_cluster_detected else ""
                        flow_tag = f"  [{controller.flow_state}]" if controller.flow_state != "normal" else ""
                        fortune = controller.generate_fortune(now)
                        fortune_tag = f"  «{fortune}»" if fortune else ""
                        coverage_tag = f"  [cov:{controller.coverage_ratio():.0%}]" if controller.total_coverage_cells_visited > 0 else ""
                        mood_tag = f"  [{controller.current_mood}]" if controller.current_mood != "neutral" else ""
                        straight_tag = f"  [straight×{now - controller.straight_line_start_time:.0f}s]" if controller.straight_line_bonus_active else ""
                        emotag = f"  [{command.mode}]" if command.mode in ("victory_dance", "happy_wiggle", "frustrated_shimmy", "confident_strut") else ""
                        grid_tag = f"  [grid:{controller.grid_coverage_ratio():.0%}]" if controller.total_grid_cells_visited > 0 else ""
                        spiral_tag = "  [spiral]" if controller.spiral_active else ""
                        smooth_tag = f"  [smooth:{controller.path_smoothness_score:.1f}]" if controller.path_smoothness_score > 0.3 else ""
                        cruise_tag = f"  [CRUISE×{controller.cruise_mode_escalation}]" if controller.cruise_mode_active else ""
                        satisfaction_tag = f"  [satisfaction:{controller.satisfaction_level:.1f}]" if controller.satisfaction_level > 0.5 else ""
                        boustrophedon_tag = "  [BOUSTROPHEDON]" if controller.boustrophedon_active else ""
                        playfulness_tag = f"  [playful:{controller.playfulness_level:.1f}]" if controller.playfulness_level > 0.5 else ""
                        room_tag = f"  [room:{controller.get_room_openness():.0%}]" if controller.room_shape_map else ""
                        momentum_tag2 = f"  [momentum:{controller.straight_line_momentum:.1f}]" if controller.straight_line_momentum > 0.2 else ""
                        topo_tag = f"  [topo:{len(controller.topological_nodes)}n/{controller.count_unvisited_nodes()}u]" if controller.topological_nodes else ""
                        area_tag = f"  [{controller.last_area_classification}]" if controller.last_area_classification != "unknown" else ""
                        vfh_tag = f"  [VFH:{len(controller.vfh_valleys)}v]" if controller.vfh_valleys else ""
                        print(f"\r[DRIVE] {arrow}  front={front_distance:5.1f}cm  hdg={command.heading:+4d}  spd={controller.current_speed:.2f}  stress={stress_ch}  [{personality}]{streak}{terrain_tag}{corner_tag}{momentum_tag}{closing_tag}{exit_tag}{recovery_tag}{shape_tag}{dead_end_pred_tag}{junction_tag}{strategy_tag}{patience_tag}{nearmiss_tag}{loop_tag}{wf_tag}{moving_tag}{hysteresis_tag}{side_corridor_tag}{cluster_tag}{flow_tag}{coverage_tag}{mood_tag}{straight_tag}{grid_tag}{spiral_tag}{smooth_tag}{cruise_tag}{satisfaction_tag}{boustrophedon_tag}{playfulness_tag}{room_tag}{momentum_tag2}{topo_tag}{area_tag}{vfh_tag}{emotag}{achievement_tag}{quip_tag}{fortune_tag}  | {emotion}   ", end='', flush=True)
                        on_inline_line = True
                        last_print_time = now

                apply_command(tbot, controller, command, now)

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

    if launch_distance_lights:
        print("\n[Y] Launching distance_lights.py ...")
        os.execl('/bin/bash', 'bash', '-c',
                 'cd ~/trilobot-python && . trilobot-env/bin/activate && python -u examples/distance_lights.py')


if __name__ == "__main__":
    main()
