def enforce_monotonicity(points):
    """
    Coerce points to pass P(t1) >= P(t2) rule for survival steps.
    KM curves are strictly monotonically non-increasing.
    
    Args:
        points: list of [x, y] pairs (data coordinates)
    Returns:
        list of [x, y] pairs with monotonicity enforced
    """
    if not points:
        return []

    processed = [list(points[0])]
    for i in range(1, len(points)):
        x, y = points[i][0], points[i][1]
        prev_y = processed[-1][1]

        # Survival probability can only decrease or stay the same
        if y > prev_y:
            y = prev_y

        processed.append([float(x), float(y)])

    return processed
