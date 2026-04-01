#!/usr/bin/env python3

import time

from trilobot import BUTTON_A, Trilobot

"""
This combines the distance-based underlights with a cautious "scared cat"
response. When something gets too close Trilobot flashes a panic color,
backs up briefly once, then stops and waits for the object to move away
before it can react again.

Stop the example by pressing button A.
"""
print("Trilobot Example: Scared Cat Distance Lights\n")


BAND1 = 20   # Distance where lights show yellow
BAND2 = 80   # Distance where lights show yellow-green
BAND3 = 100  # Distance where lights show green

YELLOW_GREEN_POINT = 192

SCARE_DISTANCE = 18
RESET_DISTANCE = 28
BACKWARD_SPEED = 0.90
BACKWARD_TIME = 0.33
LOOP_DELAY = 0.08

PANIC_ON = (255, 48, 0)
PANIC_OFF = (64, 0, 0)

tbot = Trilobot()


def colour_from_distance(distance):
    """Return an RGB color from green to red based on the measured distance."""
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


def panic_retreat():
    for colour in (PANIC_ON, PANIC_OFF, PANIC_ON):
        tbot.fill_underlighting(colour)
        time.sleep(0.08)

    tbot.backward(BACKWARD_SPEED)
    time.sleep(BACKWARD_TIME)
    tbot.stop()


armed = True

try:
    while not tbot.read_button(BUTTON_A):
        distance = tbot.read_distance(timeout=25, samples=1)
        print("Distance is {:.1f} cm".format(distance))

        tbot.fill_underlighting(colour_from_distance(distance))

        if armed and 0 < distance <= SCARE_DISTANCE:
            panic_retreat()
            armed = False
        elif not armed and distance >= RESET_DISTANCE:
            armed = True

        time.sleep(LOOP_DELAY)
finally:
    tbot.stop()
    tbot.clear_underlighting()
