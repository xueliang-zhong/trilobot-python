# Trilobot

A mid-level robot learning platform aimed at the Raspberry Pi SBC range. Learn more - https://shop.pimoroni.com/products/trilobot

[![Build Status](https://img.shields.io/github/actions/workflow/status/pimoroni/trilobot-python/test.yml?branch=main)](https://github.com/pimoroni/trilobot-python/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/pimoroni/trilobot-python/badge.svg?branch=main)](https://coveralls.io/github/pimoroni/trilobot-python?branch=main)
[![PyPi Package](https://img.shields.io/pypi/v/trilobot.svg)](https://pypi.python.org/pypi/trilobot)
[![Python Versions](https://img.shields.io/pypi/pyversions/trilobot.svg)](https://pypi.python.org/pypi/trilobot)

## Where to buy

* [https://shop.pimoroni.com/products/trilobot?variant=39594077093971](https://shop.pimoroni.com/products/trilobot?variant=39594077093971)

# Installing

We'd recommend using this library with Raspberry Pi OS Bookworm or later. It requires Python ≥3.7.

## Full install (recommended):

We've created an easy installation script that will install all pre-requisites and get you up and running with minimal efforts. To run it, fire up Terminal which you'll find in Menu -> Accessories -> Terminal
on your Raspberry Pi desktop, as illustrated below:

![Finding the terminal](http://get.pimoroni.com/resources/github-repo-terminal.png)

In the new terminal window type the commands exactly as it appears below (check for typos) and follow the on-screen instructions:

```bash
git clone https://github.com/pimoroni/trilobot-python
cd trilobot-python
./install.sh
```

**Note** Libraries will be installed in the "pimoroni" virtual environment, you will need to activate it to run examples:

```
source ~/.virtualenvs/pimoroni/bin/activate
```

## Development:

If you want to contribute, or like living on the edge of your seat by having the latest code, you can install the development version like so:

```bash
git clone https://github.com/pimoroni/trilobot-python
cd trilobot-python
./install.sh --unstable
```

## Install stable library from PyPi and configure manually

* Set up a virtual environment: `python3 -m venv --system-site-packages $HOME/.virtualenvs/pimoroni`
* Switch to the virtual environment: `source ~/.virtualenvs/pimoroni/bin/activate`
* Install the library: `pip install trilobot`

In some cases you may need to us `sudo` or install pip with: `sudo apt install python3-pip`.

This will not make any configuration changes, so you may also need to enable:

* i2c: `sudo raspi-config nonint do_i2c 0`

You can optionally run `sudo raspi-config` or the graphical Raspberry Pi Configuration UI to enable interfaces.

Some of the examples have additional dependencies. You can install them with:

```bash
pip install -r requirements-examples.txt
```

# Examples and Usage

There are many examples to get you started with your Trilobot. With the library installed on your Raspberry Pi, these can be found in the `~/Pimoroni/trilobot/examples` directory. Details about what each one does can be found in the [examples readme](../examples/README.md).

To take Trilobot further, the full API is described in the [library readme](/library/trilobot/README.md)
