import importlib.util
import math
import sys
import unittest
from pathlib import Path


def load_autonomous_car_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "autonomous_car.py"
    spec = importlib.util.spec_from_file_location("autonomous_car_example", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AutonomousCarExampleTests(unittest.TestCase):
    def test_should_scan_when_front_is_invalid(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        self.assertTrue(controller.should_scan(front_distance=0.0, now=10.0))

    def test_select_heading_prefers_open_sector_when_front_blocked(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        heading = controller.select_heading(
            scan={-80: 18.0, -45: 24.0, 0: 14.0, 45: 78.0, 80: 66.0},
            front_distance=14.0,
        )

        self.assertEqual(heading, 45)

    def test_select_heading_breaks_ties_away_from_recent_turn_habit(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.recent_turns.extend([1, 1, 1, 1])

        heading = controller.select_heading(
            scan={-80: 55.0, -45: 70.0, 0: 20.0, 45: 70.0, 80: 55.0},
            front_distance=20.0,
        )

        self.assertEqual(heading, -45)

    def test_select_heading_prefers_supported_corridor_over_isolated_far_echo(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        heading = controller.select_heading(
            scan={-80: 120.0, -45: 42.0, 0: 26.0, 45: 78.0, 80: 76.0},
            front_distance=26.0,
        )

        self.assertEqual(heading, 45)

    def test_select_heading_prefers_interpolated_gap_heading_in_wide_opening(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        heading = controller.select_heading(
            scan={-80: 20.0, -45: 25.0, 0: 40.0, 45: 90.0, 80: 88.0},
            front_distance=40.0,
        )

        self.assertEqual(heading, 62)

    def test_boredom_exploration_can_choose_interpolated_gap_heading(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        heading = controller.apply_boredom_exploration(
            scan={-80: 20.0, -45: 25.0, 0: 40.0, 45: 90.0, 80: 88.0},
            current_heading=0,
        )

        self.assertEqual(heading, 62)

    def test_plan_requests_escape_when_path_is_dangerously_close(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 62.0, -45: 58.0, 0: 10.0, 45: 30.0, 80: 18.0}
        controller.last_scan_time = 3.0

        command = controller.plan(front_distance=10.0, now=3.1)

        self.assertEqual(command.mode, "escape")
        self.assertEqual(command.heading, -80)
        self.assertLess(command.left_speed, 0.0)
        self.assertLess(command.right_speed, 0.0)

    def test_plan_escape_prefers_supported_exit_over_isolated_far_echo(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 90.0, -45: 18.0, 0: 10.0, 45: 52.0, 80: 66.0}
        controller.last_scan_time = 3.0

        command = controller.plan(front_distance=10.0, now=3.1)

        self.assertEqual(command.mode, "escape")
        self.assertEqual(command.heading, 80)
        self.assertLess(command.left_speed, 0.0)
        self.assertLess(command.right_speed, 0.0)

    def test_plan_curves_toward_open_space_when_front_is_clear(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 35.0, -45: 48.0, 0: 82.0, 45: 110.0, 80: 92.0}
        controller.last_scan_time = 5.0

        command = controller.plan(front_distance=82.0, now=5.1)

        self.assertEqual(command.mode, "drive")
        self.assertEqual(command.heading, 45)
        self.assertGreater(command.left_speed, command.right_speed)

    def test_plan_records_escape_time_when_danger_detected(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 62.0, -45: 58.0, 0: 10.0, 45: 30.0, 80: 18.0}
        controller.last_scan_time = 3.0

        self.assertEqual(len(controller.stuck_escape_times), 0)
        command = controller.plan(front_distance=10.0, now=3.1)

        self.assertEqual(command.mode, "escape")
        self.assertEqual(len(controller.stuck_escape_times), 1)
        self.assertAlmostEqual(controller.stuck_escape_times[0], 3.1)

    def test_is_stuck_returns_true_after_enough_escapes_in_window(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(stuck_window_s=3.0, stuck_escape_count=4)
        controller = module.AutonomousCarController(config)

        now = 100.0
        for t in [98.5, 99.0, 99.5, 100.0]:
            controller.stuck_escape_times.append(t)

        self.assertTrue(controller.is_stuck(now))

    def test_is_stuck_returns_false_when_escapes_are_outside_window(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(stuck_window_s=3.0, stuck_escape_count=4)
        controller = module.AutonomousCarController(config)

        now = 100.0
        # Four escapes but the first three are outside the 3-second window
        for t in [90.0, 91.0, 92.0, 99.5]:
            controller.stuck_escape_times.append(t)

        self.assertFalse(controller.is_stuck(now))

    def test_plan_uses_open_space_speed_when_all_distances_are_large(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(
            open_space_distance=70.0,
            open_space_speed=0.82,
            cruise_speed=0.62,
            cruise_distance=55.0,
        )
        controller = module.AutonomousCarController(config)
        # All scan distances exceed open_space_distance
        controller.last_scan = {-80: 80.0, -45: 90.0, 0: 100.0, 45: 95.0, 80: 85.0}
        controller.last_scan_time = 10.0
        # Pre-warm speed to simulate steady-state (speed smoothing starts from 0)
        controller.current_speed = config.open_space_speed

        command = controller.plan(front_distance=100.0, now=10.1)

        self.assertEqual(command.mode, "drive")
        # Speed should be boosted: both wheel speeds should exceed cruise_speed
        self.assertGreater(command.left_speed, config.cruise_speed)
        self.assertGreater(command.right_speed, config.cruise_speed)

    def test_plan_does_not_use_open_space_speed_when_some_distances_are_small(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(
            open_space_distance=70.0,
            open_space_speed=0.82,
            cruise_speed=0.62,
            cruise_distance=55.0,
        )
        controller = module.AutonomousCarController(config)
        # One scan distance is below open_space_distance; use symmetric side values to ensure heading=0
        controller.last_scan = {-80: 75.0, -45: 75.0, 0: 60.0, 45: 75.0, 80: 75.0}
        controller.last_scan_time = 10.0
        # Pre-warm speed to simulate steady-state (speed smoothing starts from 0)
        controller.current_speed = config.cruise_speed

        command = controller.plan(front_distance=100.0, now=10.1)

        self.assertEqual(command.mode, "drive")
        # Speed should not be boosted beyond cruise_speed (both wheels at cruise or less)
        self.assertLessEqual(command.left_speed, config.open_space_speed)
        self.assertLessEqual(command.right_speed, config.open_space_speed)
        # At least one side should not exceed cruise_speed significantly
        max_expected = config.cruise_speed * (1.0 + config.steer_gain)
        self.assertLessEqual(command.left_speed, max_expected + 1e-9)
        # More discriminating assertion: with heading=0, left_speed should equal right_speed and equal cruise_speed
        self.assertAlmostEqual(command.left_speed, config.cruise_speed, places=5)


    def test_plan_speed_ramps_up_gradually(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(
            open_space_distance=70.0,
            open_space_speed=0.82,
            cruise_distance=55.0,
            speed_accel_rate=0.10,
        )
        controller = module.AutonomousCarController(config)
        self.assertEqual(controller.current_speed, 0.0)
        # All scan distances clear and symmetric -> open_space_speed=0.82 is the target
        # Using symmetric sides so side-proximity correction does not affect left/right balance
        controller.last_scan = {-80: 85.0, -45: 90.0, 0: 100.0, 45: 90.0, 80: 85.0}
        controller.last_scan_time = 10.0

        command = controller.plan(front_distance=80.0, now=10.1)

        self.assertEqual(command.mode, "drive")
        # Speed should move toward target by at most speed_accel_rate from 0
        self.assertAlmostEqual(command.left_speed, command.right_speed, places=5)
        self.assertLessEqual(command.left_speed, config.speed_accel_rate + 1e-9)
        self.assertGreater(controller.current_speed, 0.0)
        self.assertLessEqual(controller.current_speed, config.speed_accel_rate)

    def test_plan_speed_resets_to_zero_on_escape(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.current_speed = 0.5
        controller.last_scan = {-80: 62.0, -45: 58.0, 0: 10.0, 45: 30.0, 80: 18.0}
        controller.last_scan_time = 3.0

        command = controller.plan(front_distance=10.0, now=3.1)

        self.assertEqual(command.mode, "escape")
        self.assertEqual(controller.current_speed, 0.0)

    def test_should_scan_uses_shorter_interval_in_open_space(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(open_space_distance=70.0, open_space_scan_s=1.2, proactive_scan_s=2.4)
        controller = module.AutonomousCarController(config)
        # All scan values above open_space_distance -> was_open=True
        controller.last_scan = {-80: 100.0, -45: 100.0, 0: 100.0, 45: 100.0, 80: 100.0}

        now = 100.0
        controller.last_scan_time = now - 1.5
        self.assertTrue(controller.should_scan(front_distance=100.0, now=now))

        controller.last_scan_time = now - 0.8
        self.assertFalse(controller.should_scan(front_distance=100.0, now=now))

    def test_should_scan_uses_normal_interval_when_not_open_space(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(open_space_distance=70.0, open_space_scan_s=1.2, proactive_scan_s=2.4)
        controller = module.AutonomousCarController(config)
        # One scan value below open_space_distance -> was_open=False
        controller.last_scan = {-80: 100.0, -45: 100.0, 0: 60.0, 45: 100.0, 80: 100.0}

        now = 100.0
        controller.last_scan_time = now - 1.5
        # 1.5s < proactive_scan_s=2.4s so should not trigger proactive scan
        self.assertFalse(controller.should_scan(front_distance=100.0, now=now))

    def test_update_scan_builds_angle_ema(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(history_alpha=0.25)
        controller = module.AutonomousCarController(config)

        scan1 = {-45: 40.0, 0: 50.0, 45: 60.0}
        controller.update_scan(scan1, now=1.0)

        # After first call, angle_ema should equal the scan values
        for angle, dist in scan1.items():
            self.assertAlmostEqual(controller.angle_ema[angle], dist)

        scan2 = {-45: 20.0, 0: 80.0, 45: 100.0}
        controller.update_scan(scan2, now=2.0)

        # After second call, verify EMA formula: new_ema = alpha * new_val + (1 - alpha) * old_ema
        alpha = 0.25
        for angle in scan1:
            expected = alpha * scan2[angle] + (1.0 - alpha) * scan1[angle]
            self.assertAlmostEqual(controller.angle_ema[angle], expected)

    def test_score_heading_biased_toward_historically_open_angle(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        # Right side (45) is historically much more open
        controller.angle_ema = {-80: 30.0, -45: 30.0, 0: 30.0, 45: 90.0, 80: 30.0}

        scan = {-80: 50.0, -45: 50.0, 0: 50.0, 45: 50.0, 80: 50.0}
        ordered_scan = tuple(sorted(scan.items()))

        index_pos45 = next(i for i, (a, _) in enumerate(ordered_scan) if a == 45)
        index_neg45 = next(i for i, (a, _) in enumerate(ordered_scan) if a == -45)

        score_pos45 = controller.score_heading(ordered_scan, index_pos45, front_distance=50.0)
        score_neg45 = controller.score_heading(ordered_scan, index_neg45, front_distance=50.0)

        self.assertGreater(score_pos45, score_neg45)

    def test_score_heading_unbiased_with_symmetric_history(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        # All EMA values equal -> no bias
        controller.angle_ema = {-80: 50.0, -45: 50.0, 0: 50.0, 45: 50.0, 80: 50.0}

        scan = {-80: 60.0, -45: 60.0, 0: 60.0, 45: 60.0, 80: 60.0}
        ordered_scan = tuple(sorted(scan.items()))

        index_pos45 = next(i for i, (a, _) in enumerate(ordered_scan) if a == 45)
        index_neg45 = next(i for i, (a, _) in enumerate(ordered_scan) if a == -45)

        score_pos45 = controller.score_heading(ordered_scan, index_pos45, front_distance=60.0)
        score_neg45 = controller.score_heading(ordered_scan, index_neg45, front_distance=60.0)

        self.assertAlmostEqual(score_pos45, score_neg45)

    def test_exploration_penalty_spreads_to_nearby_headings_without_mutating_memory(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.exploration_memory = {45: 1.0}

        exact_penalty = controller.exploration_penalty(45)
        nearby_penalty = controller.exploration_penalty(62)

        self.assertEqual(controller.exploration_memory, {45: 1.0})
        self.assertLess(exact_penalty, nearby_penalty)
        self.assertLess(nearby_penalty, 0.0)

    def test_motion_colour_reflects_heading_and_drive_state(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        left_colour = controller.motion_colour_for("drive", heading=-45, speed=0.6, front_distance=80.0)
        straight_colour = controller.motion_colour_for("drive", heading=0, speed=0.6, front_distance=80.0)
        right_colour = controller.motion_colour_for("drive", heading=45, speed=0.6, front_distance=80.0)
        caution_colour = controller.motion_colour_for("drive", heading=0, speed=0.2, front_distance=10.0)
        escape_colour = controller.motion_colour_for("escape", heading=-80, speed=0.0, front_distance=8.0)

        self.assertGreater(left_colour[0], left_colour[1])
        self.assertGreater(left_colour[0], left_colour[2])
        self.assertGreater(straight_colour[1], straight_colour[0])
        self.assertGreater(straight_colour[1], straight_colour[2])
        self.assertGreater(right_colour[2], right_colour[0])
        self.assertGreater(right_colour[2], right_colour[1])
        self.assertNotEqual(straight_colour, caution_colour)
        self.assertEqual(escape_colour, (255, 48, 0))

    def test_plan_triggers_peek_pounce_toward_clear_side(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(
            speed_accel_rate=1.0,
            speed_decel_rate=1.0,
        )
        controller = module.AutonomousCarController(config)
        now = 50.0
        controller.last_scan = {-80: 22.0, -45: 30.0, 0: 26.0, 45: 98.0, 80: 94.0}
        controller.last_scan_time = now - 0.1

        command = controller.plan(front_distance=26.0, now=now)

        self.assertEqual(command.mode, "peek_pounce")
        self.assertEqual(command.heading, 45)
        self.assertGreater(command.left_speed, command.right_speed)
        self.assertAlmostEqual(controller.pounce_heading, 45)
        self.assertGreater(controller.pounce_until, now)

    def test_plan_keeps_committed_pounce_heading_until_window_expires(self):
        module = load_autonomous_car_module()
        config = module.AutonomousCarConfig(
            speed_accel_rate=1.0,
            speed_decel_rate=1.0,
        )
        controller = module.AutonomousCarController(config)
        now = 80.0
        controller.last_scan = {-80: 22.0, -45: 30.0, 0: 26.0, 45: 98.0, 80: 94.0}
        controller.last_scan_time = now - 0.1

        first = controller.plan(front_distance=26.0, now=now)
        self.assertEqual(first.mode, "peek_pounce")

        controller.last_scan = {-80: 94.0, -45: 98.0, 0: 26.0, 45: 30.0, 80: 22.0}
        second = controller.plan(front_distance=26.0, now=now + config.pounce_commit_s / 2.0)

        self.assertEqual(second.mode, "peek_pounce")
        self.assertEqual(second.heading, 45)
        self.assertGreater(second.left_speed, second.right_speed)

    def test_side_correction_nudges_away_from_left_wall(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        now = 100.0
        # Scan chosen so heading=0 wins but -45 (left side) < 45 (right side)
        controller.last_scan = {-80: 20.0, -45: 30.0, 0: 80.0, 45: 55.0, 80: 20.0}
        controller.last_scan_time = now - 0.1
        # Pre-warm front_history so median = 80.0
        for _ in range(5):
            controller.front_history.append(80.0)

        command = controller.plan(front_distance=80.0, now=now)

        self.assertEqual(command.mode, "drive")
        self.assertEqual(command.heading, 0)
        # left wall closer (-45=30 < 45=55) → steer right (positive) → left_speed > right_speed
        self.assertGreater(command.left_speed, command.right_speed)

    def test_side_correction_nudges_away_from_right_wall(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        now = 100.0
        # Symmetric mirror: right wall closer (45=30 < -45=55)
        controller.last_scan = {-80: 20.0, -45: 55.0, 0: 80.0, 45: 30.0, 80: 20.0}
        controller.last_scan_time = now - 0.1
        # Pre-warm front_history so median = 80.0
        for _ in range(5):
            controller.front_history.append(80.0)

        command = controller.plan(front_distance=80.0, now=now)

        self.assertEqual(command.mode, "drive")
        self.assertEqual(command.heading, 0)
        # right wall closer (45=30 < -45=55) → steer left (negative) → right_speed > left_speed
        self.assertGreater(command.right_speed, command.left_speed)

    def test_side_correction_absent_when_turning(self):
        module = load_autonomous_car_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        now = 100.0
        # Scan that causes heading=45 (front=60 is clear, right side much more open)
        controller.last_scan = {-80: 30.0, -45: 30.0, 0: 60.0, 45: 100.0, 80: 90.0}
        controller.last_scan_time = now - 0.1

        command = controller.plan(front_distance=60.0, now=now)

        self.assertEqual(command.mode, "drive")
        # heading should be 45, so abs(heading)=45 >= 30 and side correction does not apply
        self.assertEqual(command.heading, 45)

    def test_apply_command_resets_scan_time_after_escape(self):
        module = load_autonomous_car_module()

        class MockTbot:
            def fill_underlighting(self, c): pass
            def backward(self, s): pass
            def turn_left(self, s): pass
            def turn_right(self, s): pass
            def stop(self): pass
            def set_motor_speeds(self, l, r): pass

        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan_time = 99.0
        escape_command = module.MotionCommand(
            mode="escape",
            left_speed=-0.45,
            right_speed=-0.45,
            heading=-80,
            colour=(255, 48, 0),
        )

        module.apply_command(MockTbot(), controller, escape_command)

        self.assertEqual(controller.last_scan_time, -math.inf)


if __name__ == "__main__":
    unittest.main()
