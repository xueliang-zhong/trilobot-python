# Current Memory — Trilobot Autonomous Car

## Generation history
| Gen | Commit | What changed | Result |
|-----|--------|--------------|--------|
| gen02 | c2d4168 | stuck detection + open-space speed boost + ultrasonic min-trigger guard | verified on Pi, 12 tests pass |
| gen03 | 2f5a301 | speed smoothing (accel/decel rates) + adaptive scan interval in open space | verified on Pi, 16 tests pass |
| gen04 | 700d0a8 | per-angle EMA openness bias (history_alpha=0.25, history_bias_gain=8.0) | verified on Pi, 19 tests pass |

## Active algorithm (gen04)
- **5-angle scan**: -80, -45, 0, 45, 80 degrees
- **Scoring**: corridor support + isolation penalty + distance advantage + edge penalty + turn habit anti-repetition + target heading memory + **EMA openness bias** (angles historically more open get score bonus proportional to their EMA vs mean)
- **Escape**: reverse then turn; if `is_stuck()` (≥4 escapes in 3 s), full 180° spin recovery
- **Speed**: caution→cruise→open_space (0.42→0.62→0.82) with ramp-limited transitions (accel=0.10, decel=0.18 per loop cycle)
- **Scan interval**: 1.2s when all angles clear (open space), 2.4s otherwise

## Key lessons
- Kill lingering GPIO processes before re-running: `ssh hayley@192.168.0.49 'sudo killall python3'` — timeout kills leave GPIO busy

## Key lessons
- `all()` on empty generator returns `True` — always materialise the list first before calling `all()`
- Ultrasonic HC-SR04 needs 60 ms min trigger interval; added to `trilobot/__init__.py`
- pigpiod servo warnings on Pi are cosmetic, code still runs
- Tests use `python -m unittest` (pytest not installed in Pi environment)

## Pi access
- SSH: `ssh hayley@192.168.0.49`
- Run: `cd ~/trilobot-python && . trilobot-env/bin/activate && python -u examples/autonomous_car.py`
