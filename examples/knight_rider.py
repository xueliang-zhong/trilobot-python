#!/usr/bin/env python3

import time

from trilobot import *

print("Trilobot Example: Knight Rider\n")

STEP_DELAY = 0.18
END_DELAY = 0.42
BRIGHT = 255
MID = 96
LOW = 24

MAPPING = [
    LIGHT_REAR_LEFT,
    LIGHT_MIDDLE_LEFT,
    LIGHT_FRONT_LEFT,
    LIGHT_FRONT_RIGHT,
    LIGHT_MIDDLE_RIGHT,
    LIGHT_REAR_RIGHT,
]


def draw_scanner(tbot, position):
    tbot.clear_underlighting(show=False)
    for offset, value in ((0, BRIGHT), (-1, MID), (1, MID), (-2, LOW), (2, LOW)):
        idx = position + offset
        if 0 <= idx < len(MAPPING):
            tbot.set_underlight(MAPPING[idx], value, 0, 0, show=False)
    tbot.show_underlighting()


def main():
    tbot = Trilobot()
    positions = list(range(len(MAPPING))) + list(range(len(MAPPING) - 2, 0, -1))

    try:
        while True:
            for position in positions:
                draw_scanner(tbot, position)
                if position in (0, len(MAPPING) - 1):
                    time.sleep(END_DELAY)
                else:
                    time.sleep(STEP_DELAY)
    except KeyboardInterrupt:
        pass
    finally:
        tbot.clear_underlighting()


if __name__ == "__main__":
    main()
