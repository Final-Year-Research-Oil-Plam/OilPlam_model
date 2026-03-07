"""
Run analyze_weighted_ranges logic on a hex .txt file (e.g. api.txt).
Returns the final day class (2d, 4d, 12d, 13d, 14d, 15d, 16d).
No tkinter; suitable for API use.
"""
from pathlib import Path

import numpy as np

# --- Class ranges (same as task.py / analyze_weighted_ranges.py) ---
RANGES = {
    "2d": {"R": ("ee", "ff"), "G": ("00", "9f"), "B": ("00", "08")},
    "4d": {"R": ("fa", "ff"), "G": ("00", "03"), "B": ("00", "8d")},
    "12d": {"R": ("00", "2b"), "G": ("00", "27"), "B": ("f7", "ff")},
    "16d": {"R": ("00", "ff"), "G": ("00", "ff"), "B": ("00", "ff")},
}

# --- Channel weights (match analyze_weighted_ranges.py) ---
WEIGHTS = {
    "2d": {"R": 0.6, "G": 0.2, "B": 0.0},
    "4d": {"R": 0.2, "G": 0.1, "B": 0.7},
    "12d": {"R": 0.7, "G": 0.9, "B": 0.7},
    "16d": {"R": 0.00, "G": 0.00, "B": 0.00},
}


def hex_to_rgb(hx: str):
    """'#rrggbb' -> (R, G, B) ints. Raises ValueError if not valid hex."""
    return tuple(int(hx[i : i + 2], 16) for i in (1, 3, 5))


def load_hex_file(path: str) -> np.ndarray:
    """Load #RRGGBB lines from a file into an (N,3) RGB array. Returns None if empty. Skips invalid lines."""
    rgb_list = []
    with open(path, "r") as f:
        for line in f:
            h = line.strip()
            if len(h) != 7 or not h.startswith("#"):
                continue
            try:
                rgb_list.append(hex_to_rgb(h))
            except (ValueError, TypeError):
                continue
    if not rgb_list:
        return None
    return np.array(rgb_list, dtype=int)


def compute_channel_coverage(rgb: np.ndarray) -> dict:
    """Per-class per-channel % coverage."""
    total_pixels = len(rgb)
    coverage = {}
    for range_name, channel_ranges in RANGES.items():
        percentages = []
        for idx, ch in enumerate(["R", "G", "B"]):
            start_hex, end_hex = channel_ranges[ch]
            start_val = int(start_hex, 16)
            end_val = int(end_hex, 16)
            ch_vals = rgb[:, idx]
            in_range = np.sum((ch_vals >= start_val) & (ch_vals <= end_val))
            perc = (in_range / total_pixels) * 100.0
            percentages.append(perc)
        coverage[range_name] = {"R": percentages[0], "G": percentages[1], "B": percentages[2]}
    return coverage


def compute_weighted_scores(coverage: dict) -> dict:
    """Weighted mean per class (divide by 3), normalized to sum to 100%."""
    raw_scores = {}
    for cls, ch_perc in coverage.items():
        w = WEIGHTS.get(cls, {"R": 0.0, "G": 0.0, "B": 0.0})
        if w["R"] == 0 and w["G"] == 0 and w["B"] == 0:
            raw_scores[cls] = 0.0
            continue
        score = (w["R"] * ch_perc["R"] + w["G"] * ch_perc["G"] + w["B"] * ch_perc["B"]) / 3.0
        raw_scores[cls] = score
    total = sum(raw_scores.values())
    if total == 0:
        return raw_scores
    return {cls: (score / total) * 100.0 for cls, score in raw_scores.items()}


def get_day_class_from_highest_score(highest_value: float) -> str:
    """When 12d has highest score: map value to 12d–16d."""
    if highest_value <= 50:
        return "12d"
    if highest_value <= 55:
        return "13d"
    if highest_value <= 60:
        return "14d"
    if highest_value <= 70:
        return "15d"
    return "16d"


def get_day_class_when_12d_not_highest(scores: dict, winning_class: str) -> str:
    """When 12d is not highest: gap >= 10 → winning class; else 12d >= 30 → 4d, else 2d.
    4d is only taken when 50% <= score_4d <= 70%; else use the next highest (2d, 12d, or 16d)."""
    score_2d = scores["2d"]
    score_4d = scores["4d"]
    score_12d = scores["12d"]
    rounded_2d = round(score_2d)
    rounded_4d = round(score_4d)
    gap = abs(rounded_2d - rounded_4d)

    def _when_4d_rejected() -> str:
        """4d was chosen but score_4d not in [50,70]: return next highest among 2d, 12d, 16d."""
        next_best = max(["2d", "12d", "16d"], key=lambda c: scores[c])
        if next_best == "12d":
            return get_day_class_from_highest_score(scores["12d"])
        return next_best

    take_4d = 50 <= score_4d <= 70

    if gap >= 10:
        if winning_class == "4d" and not take_4d:
            return _when_4d_rejected()
        return winning_class
    if score_12d >= 30:
        if take_4d:
            return "4d"
        return _when_4d_rejected()
    return "2d"


def run(hex_file_path: str) -> dict:
    """
    Run full analyze_weighted_ranges logic on a hex .txt file.
    Returns {"final_date": str, "winning_class": str, "scores": dict} or {"final_date": None, "error": str}.
    """
    path = Path(hex_file_path)
    if not path.exists():
        print("[date flow] day_from_hex: file not found:", hex_file_path)
        return {"final_date": None, "winning_class": None, "scores": None, "error": "File not found"}
    if not path.is_file():
        print("[date flow] day_from_hex: not a file:", hex_file_path)
        return {"final_date": None, "winning_class": None, "scores": None, "error": "Not a file"}
    try:
        rgb = load_hex_file(str(path))
    except OSError as e:
        print("[date flow] day_from_hex: read error:", e)
        return {"final_date": None, "winning_class": None, "scores": None, "error": str(e)}
    if rgb is None:
        print("[date flow] day_from_hex: no valid #RRGGBB lines in file")
        return {"final_date": None, "winning_class": None, "scores": None, "error": "No valid #RRGGBB lines"}
    total_pixels = len(rgb)
    print("[date flow] hex file loaded: %d pixels (RGB rows)" % total_pixels)
    coverage = compute_channel_coverage(rgb)
    scores = compute_weighted_scores(coverage)
    winning_class = max(["2d", "4d", "12d", "16d"], key=lambda c: scores[c])
    highest_value = scores[winning_class]

    # --- Raw weighted scores (for printing) ---
    raw_scores = {}
    for cls in ["2d", "4d", "12d", "16d"]:
        w = WEIGHTS.get(cls, {"R": 0.0, "G": 0.0, "B": 0.0})
        c = coverage.get(cls, {"R": 0, "G": 0, "B": 0})
        raw_scores[cls] = (w["R"] * c["R"] + w["G"] * c["G"] + w["B"] * c["B"]) / 3.0
    sum_raw = sum(raw_scores.values())

    # --- Print day logic (same as workspace image_to_day_class) ---
    print("[date flow] -------- WHAT HAPPENED (date logic) --------")
    print("[date flow] FORMULA: Channel coverage = (pixels in range) / total_pixels × 100")
    print("[date flow] Ranges (hex) per class:")
    for cls in ["2d", "4d", "12d", "16d"]:
        rng = RANGES[cls]
        print("[date flow]   %s: R=[0x%s-0x%s] G=[0x%s-0x%s] B=[0x%s-0x%s]" % (
            cls, rng["R"][0], rng["R"][1], rng["G"][0], rng["G"][1], rng["B"][0], rng["B"][1]))
    print("[date flow] 1. Channel coverage (result) total_pixels = %d" % total_pixels)
    for cls in ["2d", "4d", "12d", "16d"]:
        c = coverage.get(cls, {})
        print("[date flow]   %s: R=%5.1f%% G=%5.1f%% B=%5.1f%%" % (
            cls, c.get("R", 0), c.get("G", 0), c.get("B", 0)))
    print("[date flow] FORMULA: raw_cls = (w_R×cov_R + w_G×cov_G + w_B×cov_B)/3; score = raw/sum(raw)×100")
    print("[date flow] 2. Weighted score (calculation)")
    for cls in ["2d", "4d", "12d", "16d"]:
        w = WEIGHTS[cls]
        c = coverage.get(cls, {})
        raw = raw_scores[cls]
        norm = (raw / sum_raw * 100.0) if sum_raw else 0.0
        print("[date flow]   %s: raw = (%.1f×%.1f + %.1f×%.1f + %.1f×%.1f)/3 = %.2f" % (
            cls, w["R"], c.get("R", 0), w["G"], c.get("G", 0), w["B"], c.get("B", 0), raw))
        print("[date flow]        score = %.2f / %.2f × 100 = %.2f%%" % (raw, sum_raw, norm))
    print("[date flow] 3. Winning class → %s (score = %.2f%%)" % (winning_class, highest_value))

    if winning_class == "12d":
        final_date = get_day_class_from_highest_score(highest_value)
        print("[date flow] 4. Rule: 12d won → map score to day: ≤50→12d 50–55→13d 55–60→14d 60–70→15d >70→16d")
        print("[date flow]    Here: score = %.1f%% → %s" % (highest_value, final_date))
    else:
        final_date = get_day_class_when_12d_not_highest(scores, winning_class)
        gap = abs(round(scores["2d"]) - round(scores["4d"]))
        print("[date flow] 4. Rule: 12d did not win → gap(2d,4d)=%d; 4d only if 50%%≤4d%%≤70%% else next highest" % gap)
        print("[date flow]    Here: winner=%s → final_date=%s" % (winning_class, final_date))
    print("[date flow] DAY CLASS: %s" % final_date)
    print("[date flow] ----------------------------------------------")

    return {"final_date": final_date, "winning_class": winning_class, "scores": scores, "error": None}
