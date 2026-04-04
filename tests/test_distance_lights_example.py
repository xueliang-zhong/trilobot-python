import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "distance_lights.py"


def load_distance_lights_module():
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

    spec = importlib.util.spec_from_file_location("distance_lights_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, FakeTrilobot


class DistanceLightsExampleTests(unittest.TestCase):
    def test_import_does_not_construct_hardware_or_run_loop(self):
        module, fake_class = load_distance_lights_module()

        self.assertTrue(hasattr(module, "main"))
        self.assertEqual(fake_class.constructed, 0)

    def test_update_underlighting_unpacks_rgb_tuple(self):
        module, _ = load_distance_lights_module()

        class FakeTbot:
            def __init__(self):
                self.calls = []

            def fill_underlighting(self, r, g, b):
                self.calls.append((r, g, b))

        tbot = FakeTbot()
        module.update_underlighting(tbot, 42.0)

        self.assertEqual(tbot.calls, [module.colour_from_distance(42.0)])


if __name__ == "__main__":
    unittest.main()
