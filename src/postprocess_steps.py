def enforce_monotonicity(points):
    """
    Coerce points to pass P(t1) >= P(t2) rule for survival steps.
    """
    if not points:
        return []
    
    processed = [points[0]]
    for i in range(1, len(points)):
        x, y = points[i]
        # Previous y value
        prev_y = processed[-1][1]
        
        # Survival prob can only decrease or stay the same
        # Assuming higher y means lower probability here, wait survival step goes down over time.
        # So curve must be monotonically decreasing.
        if y > prev_y:
            y = prev_y
            
        processed.append((x, y))
    
    return processed
