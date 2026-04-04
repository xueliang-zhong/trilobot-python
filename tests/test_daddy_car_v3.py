import importlib.util
import math
import sys
import unittest
from pathlib import Path


def load_daddy_car_v3_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "daddy_car_v3.py"
    spec = importlib.util.spec_from_file_location("daddy_car_v3_example", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DaddyCarV3Tests(unittest.TestCase):
    def test_defaults_disable_nonessential_motion_modes(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        controller.total_grid_cells_visited = 10
        self.assertFalse(controller.should_activate_room_sweep(now=100.0))
        self.assertFalse(controller.should_activate_boustrophedon(now=100.0))
        self.assertFalse(controller.should_activate_edge_patrol(now=100.0))
        self.assertFalse(controller.should_activate_wander(now=100.0))
        self.assertEqual(controller.maybe_start_spiral_exploration(heading=15, front_distance=120.0, now=100.0), 15)

        controller.trigger_emotional_move("victory_dance", now=100.0)
        self.assertFalse(controller.emotional_move_active)

    def test_select_heading_prefers_straight_when_front_lane_is_fast_and_safe(self):
        module = load_daddy_car_v3_module()
        config = module.AutonomousCarConfig(
            straight_preference_gain=28.0,
            straight_preference_distance=85.0,
        )
        controller = module.AutonomousCarController(config)

        heading = controller.select_heading(
            scan={-80: 118.0, -45: 126.0, 0: 121.0, 45: 128.0, 80: 117.0},
            front_distance=121.0,
            now=50.0,
        )

        self.assertEqual(heading, 0)

    def test_perform_scan_returns_zeroes_when_hardware_calls_fail(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        class BrokenTbot:
            def set_servo_angle(self, angle):
                raise RuntimeError("servo failure")

            def read_distance(self, timeout, samples):
                raise RuntimeError("sensor failure")

        scan = module.perform_scan(BrokenTbot(), controller)

        self.assertEqual(set(scan), set(controller.config.scan_angles))
        self.assertTrue(all(scan[angle] == 0.0 for angle in controller.config.scan_angles))
        self.assertEqual(controller.last_scan, scan)

    def test_apply_command_swallows_drive_hardware_failures_and_forces_stop(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())

        class BrokenDriveTbot:
            def __init__(self):
                self.stop_calls = 0

            def fill_underlighting(self, *args, **kwargs):
                pass

            def clear_underlighting(self):
                pass

            def set_motor_speeds(self, left, right):
                raise RuntimeError("motor driver failure")

            def stop(self):
                self.stop_calls += 1

        command = module.MotionCommand(
            mode="drive",
            left_speed=0.6,
            right_speed=0.6,
            heading=0,
            colour=(0, 255, 0),
        )
        tbot = BrokenDriveTbot()

        module.apply_command(tbot, controller, command, now=10.0)

        self.assertGreaterEqual(tbot.stop_calls, 1)

    def test_safe_lighting_proxy_clamps_invalid_rgb_values(self):
        module = load_daddy_car_v3_module()
        recorded = []

        class StrictLights:
            def set_underlight(self, led, r, g, b, show=True):
                if not all(0 <= value <= 255 for value in (r, g, b)):
                    raise ValueError("rgb out of range")
                recorded.append((led, r, g, b, show))

            def fill_underlighting(self, r, g, b):
                if not all(0 <= value <= 255 for value in (r, g, b)):
                    raise ValueError("rgb out of range")
                recorded.append(("fill", r, g, b))

            def show_underlighting(self):
                recorded.append(("show",))

        proxy = module._SafeLightingProxy(StrictLights())
        proxy.set_underlight(1, 999, -25, 128)
        proxy.fill_underlighting(400, 20, -10)
        proxy.fill_underlighting_clamp_rgb_tuple((1.5, -0.2, 0.5))
        proxy.show_underlighting()

        self.assertIn((1, 255, 0, 128, True), recorded)
        self.assertIn(("fill", 255, 20, 0), recorded)
        self.assertIn(("fill", 255, 0, 128), recorded)
        self.assertIn(("show",), recorded)


if __name__ == "__main__":
    unittest.main()
