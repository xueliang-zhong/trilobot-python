import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "scared_cat_distance_lights.py"


def load_scared_cat_module():
    fake_trilobot = types.ModuleType("trilobot")
    fake_trilobot.BUTTON_A = 0

    class FakeTrilobot:
        constructed = 0

        def __init__(self):
            FakeTrilobot.constructed += 1

        def read_button(self, button):
            return True

    fake_trilobot.Trilobot = FakeTrilobot
    sys.modules["trilobot"] = fake_trilobot

    spec = importlib.util.spec_from_file_location("scared_cat_distance_lights_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, FakeTrilobot


class ScaredCatDistanceLightsExampleTests(unittest.TestCase):
    def test_import_does_not_construct_hardware_or_run_loop(self):
        module, fake_class = load_scared_cat_module()

        self.assertTrue(hasattr(module, "main"))
        self.assertEqual(fake_class.constructed, 0)

    def test_decide_reaction_uses_simple_fight_or_flight_strategy(self):
        module, _ = load_scared_cat_module()

        self.assertEqual(module.decide_reaction(10.0, armed=True), "flight")
        self.assertEqual(module.decide_reaction(22.0, armed=True), "charge")
        self.assertEqual(module.decide_reaction(60.0, armed=True), "idle")
        self.assertEqual(module.decide_reaction(10.0, armed=False), "idle")

    def test_charge_action_uses_white_lights(self):
        module, _ = load_scared_cat_module()
        events = []

        class FakeTbot:
            def fill_underlighting(self, *rgb):
                events.append(("lights", rgb))

            def forward(self, speed):
                events.append(("forward", speed))

            def stop(self):
                events.append(("stop",))

        original_sleep = module.time.sleep
        module.time.sleep = lambda _: None
        try:
            module.charge_forward(FakeTbot())
        finally:
            module.time.sleep = original_sleep

        self.assertIn(("lights", module.CHARGE_ON), events)
        self.assertIn(("forward", module.CHARGE_SPEED), events)
        self.assertIn(("stop",), events)


if __name__ == "__main__":
    unittest.main()
