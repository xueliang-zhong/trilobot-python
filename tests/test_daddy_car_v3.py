import importlib.util
import math
import re
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

    def test_shutdown_tbot_stops_and_cleans_up_without_raising(self):
        module = load_daddy_car_v3_module()
        events = []

        class MostlyBrokenTbot:
            def stop(self):
                events.append("stop")

            def clear_underlighting(self):
                events.append("clear")

            def set_servo_angle(self, angle):
                events.append(("servo", angle))
                raise RuntimeError("servo already gone")

            def disable_servo(self):
                events.append("disable")

            def cleanup(self):
                events.append("cleanup")

        module.shutdown_tbot(MostlyBrokenTbot())

        self.assertIn("stop", events)
        self.assertIn("clear", events)
        self.assertIn("cleanup", events)

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

    def test_plan_escape_retreats_more_aggressively_for_immediate_blocker(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 40.0, -45: 35.0, 0: 6.0, 45: 38.0, 80: 42.0}
        controller.last_scan_time = 3.0

        command = controller.plan(front_distance=6.0, now=3.1)

        self.assertEqual(command.mode, "escape")
        self.assertLess(command.left_speed, -controller.config.escape_reverse_speed)
        self.assertLess(command.right_speed, -controller.config.escape_reverse_speed)

    def test_plan_brave_push_keeps_forward_charge_behavior(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 70.0, -45: 68.0, 0: 20.0, 45: 72.0, 80: 74.0}
        controller.last_scan_time = 5.0
        controller.last_push_time = -math.inf

        command = controller.plan(front_distance=20.0, now=5.1)

        self.assertEqual(command.mode, "brave_push")
        self.assertGreater(command.left_speed, 0.0)
        self.assertGreater(command.right_speed, 0.0)

    def test_plan_triggers_playful_charge_when_someone_rapidly_approaches(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 72.0, -45: 65.0, 0: 45.0, 45: 66.0, 80: 73.0}
        controller.last_scan_time = 4.0
        controller.track_approach_rate(45.0, 4.0)
        controller.track_approach_rate(32.0, 4.4)
        controller.track_approach_rate(24.0, 4.8)
        controller.last_scan[0] = 24.0

        command = controller.plan(front_distance=24.0, now=4.9)

        self.assertEqual(command.mode, "playful_charge")
        self.assertGreater(command.left_speed, 0.0)
        self.assertGreater(command.right_speed, 0.0)

    def test_plan_triggers_playful_charge_more_easily_for_close_playful_approach(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 80.0, -45: 52.0, 0: 38.0, 45: 54.0, 80: 78.0}
        controller.last_scan_time = 6.0
        controller.track_approach_rate(46.0, 6.0)
        controller.track_approach_rate(41.0, 6.4)
        controller.track_approach_rate(38.0, 6.8)
        controller.last_scan[0] = 38.0

        command = controller.plan(front_distance=38.0, now=6.9)

        self.assertEqual(command.mode, "playful_charge")

    def test_plan_avoids_playful_charge_when_sides_are_too_pinched(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 42.0, -45: 18.0, 0: 24.0, 45: 17.0, 80: 41.0}
        controller.last_scan_time = 8.0
        controller.track_approach_rate(42.0, 8.0)
        controller.track_approach_rate(30.0, 8.3)
        controller.track_approach_rate(24.0, 8.6)
        controller.last_scan[0] = 24.0

        command = controller.plan(front_distance=24.0, now=8.7)

        self.assertNotEqual(command.mode, "playful_charge")

    def test_describe_lights_for_playful_charge_uses_white_charge_theme(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        command = module.MotionCommand("playful_charge", 0.75, 0.68, 12, (255, 255, 255))

        lights = module._describe_lights(command, controller, 24.0)

        self.assertIn("white", lights.lower())
        self.assertIn("charge", lights.lower())

    def test_update_position_awareness_marks_visual_stuck_when_scene_does_not_change(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.current_speed = 0.62
        controller.last_scan = {-45: 34.0, 0: 32.0, 45: 33.0}

        controller.update_position_awareness(front_distance=32.0, now=1.0)
        controller.update_position_awareness(front_distance=31.5, now=1.7)
        controller.update_position_awareness(front_distance=31.4, now=2.5)

        self.assertTrue(controller.is_visually_stuck(2.5))

    def test_update_position_awareness_detects_stuck_despite_small_sensor_wobble(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.current_speed = 0.62

        snapshots = [
            (1.0, {-80: 36.0, -45: 34.0, 0: 32.0, 45: 33.0, 80: 35.0}, 32.0),
            (1.7, {-80: 38.1, -45: 36.2, 0: 33.9, 45: 35.1, 80: 37.0}, 33.9),
            (2.5, {-80: 36.4, -45: 34.4, 0: 32.3, 45: 33.2, 80: 35.4}, 32.3),
            (3.3, {-80: 37.9, -45: 35.8, 0: 33.4, 45: 34.7, 80: 36.8}, 33.4),
        ]

        for now, scan, front in snapshots:
            controller.last_scan = scan
            controller.update_position_awareness(front_distance=front, now=now)

        self.assertTrue(controller.is_visually_stuck(3.3))

    def test_plan_uses_recovery_when_visually_stuck_even_without_front_danger(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.current_speed = 0.58
        controller.last_scan = {-80: 38.0, -45: 34.0, 0: 32.0, 45: 33.0, 80: 36.0}
        controller.last_scan_time = 4.0
        controller.update_position_awareness(front_distance=32.0, now=4.2)
        controller.update_position_awareness(front_distance=31.6, now=5.0)
        controller.update_position_awareness(front_distance=31.5, now=5.8)

        command = controller.plan(front_distance=31.5, now=5.9)

        self.assertEqual(command.mode, "escape")
        self.assertNotEqual(command.left_speed, command.right_speed)
        self.assertEqual(controller.recovery_stage, 1)

    def test_render_dashboard_includes_compact_see_and_think_sections(self):
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
        self.assertNotIn("┌─ LIGHTS", dashboard)
        self.assertIn("open road ahead", dashboard)
        self.assertIn("SCAN refreshed", dashboard)
        self.assertIn("\x1b[", dashboard)
        self.assertIn("LEFT", dashboard)
        self.assertIn("AHEAD", dashboard)
        self.assertIn("RIGHT", dashboard)
        self.assertIn("drive lane", dashboard)
        self.assertIn("front", dashboard)
        self.assertIn("steer", dashboard)
        self.assertIn("rear", dashboard)

    def test_render_dashboard_keeps_see_box_visibly_compact(self):
        module = load_daddy_car_v3_module()
        controller = module.AutonomousCarController(module.AutonomousCarConfig())
        controller.last_scan = {-80: 74.3, -45: 82.0, 0: 118.1, 45: 86.2, 80: 68.5}
        controller.current_speed = 0.66
        command = module.MotionCommand("drive", 0.66, 0.61, 13, (32, 200, 180))

        dashboard = module.render_tui_dashboard(
            controller=controller,
            command=command,
            front_distance=118.1,
            emotion="feel=curious",
            quip="...",
            now=22.0,
            follow_mode=False,
            light_description="feel=curious   | F:teal(118cm open!)      M:blue(turning-R)    R:teal(cruising)",
            transient_message="scan refreshed",
        )

        visible_lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in dashboard.splitlines()]
        see_top = next(line for line in visible_lines if line.startswith("┌─ SEE"))
        self.assertLess(len(see_top), 130)

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
