from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP


def _decimal_probability(value, name, closed=True):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a probability")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a probability") from None
    valid = result.is_finite() and (
        Decimal(0) <= result <= Decimal(1)
        if closed
        else Decimal(0) < result < Decimal(1)
    )
    if not valid:
        raise ValueError(f"{name} must lie {'in [0, 1]' if closed else 'strictly between 0 and 1'}")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def promotion_budget(T, eta):
    _positive_integer(T, "T")
    value = _decimal_probability(eta, "eta")
    return int((Decimal(T) * value).to_integral_value(rounding=ROUND_FLOOR))


def calibrate(scores: list[float], alpha: float, T: int) -> dict:
    _positive_integer(T, "T")
    alpha_value = _decimal_probability(alpha, "alpha", closed=False)
    if not isinstance(scores, list) or not scores:
        raise ValueError("scores must contain at least one calibration score")
    counts = []
    for score in scores:
        scaled = _decimal_probability(score, "score") * Decimal(T)
        count = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
        if abs(scaled - Decimal(count)) > Decimal("1e-9"):
            raise ValueError("calibration scores must be integer promotion counts divided by T")
        counts.append(count)
    n = len(counts)
    index = int((Decimal(n + 1) * (Decimal(1) - alpha_value)).to_integral_value(rounding=ROUND_CEILING))
    threshold = sorted(counts + [T])[index - 1]
    return {"q_alpha": threshold / T, "h_alpha": threshold, "N": n}
