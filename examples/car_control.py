#!/usr/bin/env python3

import time
import math
from trilobot import *
from trilobot import controller_mappings

"""
An advanced example of how Trilobot can be remote controlled using a controller or gamepad.
This will require one of the supported controllers to already be paired to your Trilobot.

At startup a list of supported controllers will be shown, with you being asked to select one.
The program will then attempt to connect to the controller, and if successful Trilobot's
underlights will illuminate with a rainbow pattern.

From there you can drive your Trilobot around like a videogame car- using the right trigger
to accelerate and the left trigger to brake. The left analog stick controls steering and
will also steer you on the spot for more accurate control.

If your controller becomes disconnected Trilobot will stop moving and show a slow red
pulsing animation on its underlights. Simply reconnect your controller and after 10 to 20
seconds, the program should find your controller again and start up again.

Support for further controllers can be added to library/trilobot/controller_mappings.py

Press CTRL + C to exit.
"""
print("Trilobot Example: Car Control\n")


SPEED_DAMPING = 0.2

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

tbot = Trilobot()

# Presents the user with an option of what controller to use
controller = controller_mappings.choose_controller()

# Attempt to connect to the created controller
controller.connect()

# Run an amination on the underlights to show a controller has been selected
for led in range(NUM_UNDERLIGHTS):
    tbot.clear_underlighting(show=False)
    tbot.set_underlight(led, RED)
    time.sleep(0.1)
    tbot.clear_underlighting(show=False)
    tbot.set_underlight(led, GREEN)
    time.sleep(0.1)
    tbot.clear_underlighting(show=False)
    tbot.set_underlight(led, BLUE)
    time.sleep(0.1)

tbot.clear_underlighting()

h = 0
v = 0
spacing = 1.0 / NUM_UNDERLIGHTS

speed_left = 0
speed_right = 0

while True:

    if not controller.is_connected():
        # Attempt to reconnect to the controller if 10 seconds have passed since the last attempt
        controller.reconnect(10, True)

    try:
        # Get the latest information from the controller. This will throw a RuntimeError if the controller connection is lost
        controller.update()
    except RuntimeError:
        # Lost contact with the controller, so disable the motors to stop Trilobot if it was moving
        tbot.disable_motors()

    if controller.is_connected():

        accel = controller.read_axis("R2")
        brake = controller.read_axis("L2")
        steer = controller.read_axis("LX") # Left = -1, Right = +1

        speed_left = accel - brake
        speed_right = accel - brake

        msl = speed_left + (steer)
        msr = speed_right + (steer * -1)

        tbot.set_left_speed(msl)
        tbot.set_right_speed(msr)
 
        # Run a rotating rainbow effect on the RGB underlights
        for led in range(NUM_UNDERLIGHTS):
            led_h = h + (led * spacing)
            led_h %= 1.0
            tbot.set_underlight_hsv(led, led_h, show=False)

        tbot.show_underlighting()

        # Advance the rotating rainbow effect
        h += 0.5 / 360
        if h >= 1.0:
            h -= 1.0

    else:
        # Run a slow red pulsing animation to show there is no controller connected
        val = (math.sin(v) / 2.0) + 0.5
        tbot.fill_underlighting(val * 127, 0, 0)
        v += math.pi / 200

    time.sleep(0.01)
