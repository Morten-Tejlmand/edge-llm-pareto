"""Integrate timestamped watts without extrapolating missing telemetry."""
import math


def integrate(samples, start, end, max_gap_s=0.5):
    result = {"run_energy_j": None, "energy_coverage": 0.0, "energy_valid": False}
    if end <= start:
        return result
    points = sorted((s["t"], s["power_w"]) for s in samples
                    if s.get("power_w") is not None and math.isfinite(s["power_w"])
                    and s["power_w"] >= 0)
    energy = covered = 0.0
    for (a, pa), (b, pb) in zip(points, points[1:]):
        lo, hi = max(a, start), min(b, end)
        if hi <= lo or b <= a or b - a > max_gap_s:
            continue
        p0 = pa + (pb - pa) * (lo - a) / (b - a)
        p1 = pa + (pb - pa) * (hi - a) / (b - a)
        energy += (p0 + p1) * 0.5 * (hi - lo)
        covered += hi - lo
    result["energy_coverage"] = covered / (end - start)
    result["energy_valid"] = covered >= (end - start) * (1 - 1e-6)
    if result["energy_valid"]:
        result["run_energy_j"] = energy
    return result
