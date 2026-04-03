# Current Memory — Trilobot Autonomous Car

## Generation history
| Gen | Commit | What changed | Result |
|-----|--------|--------------|--------|
| gen02 | c2d4168 | stuck detection + open-space speed boost + ultrasonic min-trigger guard | verified on Pi, 12 tests pass |
| gen03 | 2f5a301 | speed smoothing (accel/decel rates) + adaptive scan interval in open space | verified on Pi, 16 tests pass |
| gen04 | 700d0a8 | per-angle EMA openness bias (history_alpha=0.25, history_bias_gain=8.0) | verified on Pi, 19 tests pass |
| gen05 | dda9982 | side-proximity correction (gain=0.10) + forced post-escape rescan | verified on Pi, 23 tests pass |
| gen06 | d2e1936 | gated gap-heading interpolation + boredom exploration reuses gap candidates | local unittest discover passed; Pi SSH was blocked in this sandbox |
| gen07 | blocked | pure exploration scoring + spread penalty for nearby headings + state-aware mood lighting | local unittest discover passed; .git writes blocked in this sandbox |

## Active algorithm (gen07)
- **5-angle scan**: -80, -45, 0, 45, 80 degrees
- **Scoring**: corridor support + isolation penalty + distance advantage + edge penalty + turn habit anti-repetition + target heading memory + EMA openness bias + pure exploration penalty with nearby-angle spread
- **Gap headings**: interpolate midpoints only when front distance is in the moderate band between caution and cruise, so wide openings can be centered without overruling blocked/open-space behavior
- **Escape**: reverse then turn; if `is_stuck()` (≥4 escapes in 3 s), full 180° spin recovery; forces immediate rescan after every escape
- **Speed**: caution→cruise→open_space (0.42→0.62→0.82) with ramp-limited transitions (accel=0.10, decel=0.18 per loop cycle)
- **Scan interval**: 1.2s when all angles clear (open space), 2.4s otherwise
- **Side correction**: when heading is near-straight (abs<30°) and front is clear, nudge steer proportionally away from closer side wall (gain=0.10)
- **Lights**: `motion_colour_for()` encodes drive heading, speed, openness, and scan confidence into color; escape stays hot red/orange and dead-end recovery stays amber

## Key lessons
- Kill lingering GPIO processes before re-running: `ssh hayley@192.168.0.49 'sudo killall python3'` — timeout kills leave GPIO busy
- `all()` on empty generator returns `True` — always materialise the list first before calling `all()`
- Ultrasonic HC-SR04 needs 60 ms min trigger interval; added to `trilobot/__init__.py`
- pigpiod servo warnings on Pi are cosmetic, code still runs
- Tests use `python -m unittest` (pytest not installed in Pi environment)
- Gap interpolation only helps if it is gated to moderate front distances; enabling it unconditionally can override stable blocked/open-space heading choices and break existing turn selection.
- Reuse gap candidates in boredom-driven exploration, but keep the main `select_heading()` path conservative by gating interpolation to the moderate-front-distance band.
- Never mutate exploration memory while scoring headings. Decay it once per decision cycle, then compute a pure proximity-weighted penalty so candidate order does not change the result.
- A single `motion_colour_for()` helper keeps LED meaning consistent: escape stays hot, recovery stays amber, and drive colours can encode heading plus confidence instead of hard-coded one-off tuples.

## Pi access
- SSH: `ssh hayley@192.168.0.49`
- Run: `cd ~/trilobot-python && . trilobot-env/bin/activate && python -u examples/autonomous_car.py`
