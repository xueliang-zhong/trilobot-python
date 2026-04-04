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

    def test_gap_width_filter_rejects_heading_when_chassis_clearance_is_too_small(self):
        module = load_daddy_car_v3_module()
        config = module.AutonomousCarConfig(
            chassis_width_cm=16.0,
            wheel_clearance_margin_cm=6.0,
            min_side_clearance_cm=13.0,
            gap_reject_penalty=40.0,
        )
        controller = module.AutonomousCarController(config)
        scan = {-80: 60.0, -45: 12.0, 0: 95.0, 45: 44.0, 80: 70.0}

        penalized = controller.apply_gap_width_filter(angle=-45, scan=scan, score=25.0)

        self.assertLess(penalized, -5.0)

    def test_select_heading_avoids_side_pinch_trap_even_when_front_is_open(self):
        module = load_daddy_car_v3_module()
        config = module.AutonomousCarConfig(
            straight_preference_gain=18.0,
            straight_preference_distance=70.0,
            chassis_width_cm=16.0,
            wheel_clearance_margin_cm=6.0,
            min_side_clearance_cm=13.0,
        )
        controller = module.AutonomousCarController(config)

        heading = controller.select_heading(
            scan={-80: 55.0, -45: 10.0, 0: 96.0, 45: 72.0, 80: 68.0},
            front_distance=96.0,
            now=75.0,
        )

        self.assertNotEqual(heading, -45)
        self.assertEqual(heading, 0)

    def test_render_dashboard_includes_see_think_and_lights_sections(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 40.0, -45: 55.0, 0: 88.0, 45: 70.0, 80: 48.0}
        controller.current_speed = 0.68
        command = module.MotionCommand("drive", 0.68, 0.61, 12, (32, 200, 180))

        dashboard = module.render_tui_dashboard(
            controller=controller,
            command=command,
            front_distance=88.0,
            emotion="feel=confident | F:teal M:green R:bright-teal",
            quip="open road ahead",
            now=12.0,
            follow_mode=False,
            transient_message="SCAN refreshed",
        )

        self.assertIn("SEE", dashboard)
        self.assertIn("THINK", dashboard)
        self.assertIn("LIGHTS", dashboard)
        self.assertIn("open road ahead", dashboard)
        self.assertIn("SCAN refreshed", dashboard)
        self.assertIn("\x1b[", dashboard)
        self.assertIn("LEFT", dashboard)
        self.assertIn("AHEAD", dashboard)
        self.assertIn("RIGHT", dashboard)
        self.assertIn("drive lane", dashboard)

    def test_render_forward_view_places_sensor_columns_in_car_pov_scene(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig(max_distance=140.0))
        scan = {-80: 25.0, -45: 45.0, 0: 110.0, 45: 75.0, 80: 30.0}

        scene = module.render_forward_view(scan, controller)

        self.assertIn("LEFT", scene)
        self.assertIn("AHEAD", scene)
        self.assertIn("RIGHT", scene)
        self.assertIn("drive lane", scene)
        self.assertIn("110.0cm", scene)
        self.assertIn("CLEAR", scene)
        self.assertIn("\x1b[38;2;", scene)

    def test_render_underlight_swatch_uses_truecolor_ansi_blocks(self):
        module = load_daddy_car_v3_module()

        swatch = module.render_underlight_swatch((12, 34, 56), label="front")

        self.assertIn("\x1b[48;2;12;34;56m", swatch)
        self.assertIn("front", swatch)


if __name__ == "__main__":
    unittest.main()
