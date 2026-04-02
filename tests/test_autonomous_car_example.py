import importlib.util
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
        # All scan distances clear -> open_space_speed=0.82 is the target
        controller.last_scan = {-80: 80.0, -45: 90.0, 0: 100.0, 45: 95.0, 80: 85.0}
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


if __name__ == "__main__":
    unittest.main()
