DECIMALS = 2
FRACTION_DECIMALS = DECIMALS + 2


def hundredths(value, unit: str | None = None) -> float | None:
    if value is None:
        return None
    return round(float(value), FRACTION_DECIMALS if unit == "fraction" else DECIMALS) + 0.0
