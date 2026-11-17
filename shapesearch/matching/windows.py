import math
from dataclasses import dataclass

from shapesearch.matching.query import SKETCH_RATE

SCALES = (0.8, 1.0, 1.25)   # a 30 s sketch can match a 24 s, 30 s or 37.5 s passage
STEP_S = 1.0                # try a new start time every second


@dataclass(frozen=True)
class Window:
    start_s: float          # where the window starts in the original audio file
    end_s: float
    scale: float            # which length it is: 0.8x, 1x or 1.25x the sketch
    start_idx: int          # position in the thinned curve (2 values per second)
    n: int                  # number of values

    @property
    def slice(self):
        """Use as curve[window.slice]; the same slice works for every feature of the track."""
        return slice(self.start_idx, self.start_idx + self.n)


def generate_windows(n_cells, usable_start_s, usable_end_s, target_cells,
                     scales=SCALES, step_s=STEP_S, rate=SKETCH_RATE):
    """Every (start, length) window for one track, as array slices.

    n_cells        length of the track's thinned curves
    target_cells   length of the sketch curve (60 for a 30 s sketch)
    Windows start on whole multiples of step_s, and only windows that lie fully
    inside the usable range (start_s to end_s) are kept, so silent intros and
    outros are never searched.
    """
    lengths, seen = [], set()
    for scale in scales:
        n = max(2, int(round(target_cells * scale)))
        if n not in seen:
            seen.add(n)
            lengths.append((scale, n))

    first = math.ceil(usable_start_s / step_s - 1e-9)
    windows = []
    k = first
    while True:
        start_s = k * step_s
        if start_s + min(n for _, n in lengths) / rate > usable_end_s + 1e-9:
            break
        start_idx = int(round(start_s * rate))
        for scale, n in lengths:
            end_s = start_s + n / rate
            if end_s <= usable_end_s + 1e-9 and start_idx + n <= n_cells:
                windows.append(Window(start_s, end_s, scale, start_idx, n))
        k += 1
    return windows