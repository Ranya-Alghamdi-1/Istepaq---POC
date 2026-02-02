import numpy as np
from dataclasses import dataclass

@dataclass
class Alerts:
    rapid_eye: bool = False
    no_look_too_long: bool = False
    repeated_look_away: bool = False

def iris_velocity(prev_xy, curr_xy, dt):
    if prev_xy is None or dt <= 1e-6:
        return 0.0
    d = np.linalg.norm(curr_xy - prev_xy)
    return float(d / dt)
