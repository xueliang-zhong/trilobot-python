#!/usr/bin/env python3

"""
This brings together the underlights and the distance sensor, using the underlights
to indicate if something is too close with red, orange, green indications.
It also prints distances on the console.

Stop the example by pressing button A.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trilobot import BUTTON_A, Trilobot


BAND1 = 20  # Distance where lights show yellow
BAND2 = 80  # Distance where lights show yellow-green
BAND3 = 100  # Distance where lights show green

YELLOW_GREEN_POINT = 192  # The amount of red to show for the mid-point between green and yellow


def colour_from_distance(distance):
    """Return a colour based on distance, fading from green at > 100cm to red at 0cm."""
    r = 0
    g = 0
    b = 0

    if distance > BAND3:
        g = 255
    elif distance > BAND2:
        band_min = BAND2
        band_max = BAND3
        r = int(YELLOW_GREEN_POINT - YELLOW_GREEN_POINT * (distance - band_min) / (band_max - band_min))
        g = 255
    elif distance > BAND1:
        band_min = BAND1
        band_max = BAND2
        r = int(255 - (255 - YELLOW_GREEN_POINT) * (distance - band_min) / (band_max - band_min))
        g = 255
    elif distance > 0:
        band_max = BAND1 * BAND1
        r = 255
        g = int(255 * distance * BAND1 / band_max)
    else:
        r = 255

    return (r, g, b)


def update_underlighting(tbot, distance):
    rgb_colour = colour_from_distance(distance)
    tbot.fill_underlighting(*rgb_colour)


def main():
    print("Trilobot Example: Distance Lights\n")

    tbot = Trilobot()
    try:
        while not tbot.read_button(BUTTON_A):
            distance = tbot.read_distance()
            print("Distance is {:.1f} cm".format(distance))
            update_underlighting(tbot, distance)
            time.sleep(0.1)
    finally:
        try:
            tbot.clear_underlighting()
        except Exception:
            pass
        try:
            tbot.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
