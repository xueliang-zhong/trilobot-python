#!/usr/bin/env python3

"""
This combines distance-based underlights with a very simple fight-or-flight
response. Trilobot will either retreat in panic when something is extremely
close, or do a short white-light charge when something is close but not yet
touching its nose.

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


BAND1 = 20
BAND2 = 80
BAND3 = 100
YELLOW_GREEN_POINT = 192

FLIGHT_DISTANCE = 14
CHARGE_DISTANCE = 26
RESET_DISTANCE = 34

BACKWARD_SPEED = 0.90
BACKWARD_TIME = 0.33
CHARGE_SPEED = 0.85
CHARGE_TIME = 0.20
LOOP_DELAY = 0.08

PANIC_ON = (255, 48, 0)
PANIC_OFF = (64, 0, 0)
CHARGE_ON = (255, 255, 255)
CHARGE_OFF = (80, 80, 80)


def colour_from_distance(distance):
    """Return an RGB colour from green to red based on the measured distance."""
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
    tbot.fill_underlighting(*colour_from_distance(distance))


def decide_reaction(distance, armed):
    if not armed or distance <= 0:
        return "idle"
    if distance <= FLIGHT_DISTANCE:
        return "flight"
    if distance <= CHARGE_DISTANCE:
        return "charge"
    return "idle"


def panic_retreat(tbot):
    for colour in (PANIC_ON, PANIC_OFF, PANIC_ON):
        tbot.fill_underlighting(*colour)
        time.sleep(0.08)

    tbot.backward(BACKWARD_SPEED)
    time.sleep(BACKWARD_TIME)
    tbot.stop()


def charge_forward(tbot):
    for colour in (CHARGE_ON, CHARGE_OFF, CHARGE_ON):
        tbot.fill_underlighting(*colour)
        time.sleep(0.05)

    tbot.forward(CHARGE_SPEED)
    time.sleep(CHARGE_TIME)
    tbot.stop()


def main():
    print("Trilobot Example: Scared Cat Distance Lights\n")

    tbot = Trilobot()
    armed = True
    try:
        while not tbot.read_button(BUTTON_A):
            distance = tbot.read_distance(timeout=25, samples=1)
            print("Distance is {:.1f} cm".format(distance))

            reaction = decide_reaction(distance, armed)
            if reaction == "flight":
                panic_retreat(tbot)
                armed = False
            elif reaction == "charge":
                charge_forward(tbot)
                armed = False
            else:
                update_underlighting(tbot, distance)
                if not armed and distance >= RESET_DISTANCE:
                    armed = True

            time.sleep(LOOP_DELAY)
    finally:
        try:
            tbot.stop()
        except Exception:
            pass
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
