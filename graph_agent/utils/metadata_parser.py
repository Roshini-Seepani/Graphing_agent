def sanitize_color(color: str) -> str:
    """
    Matplotlib doesn't accept color names with spaces.
    'light green' -> 'lightgreen', 'dark blue' -> 'darkblue', etc.
    """
    if color and isinstance(color, str):
        return color.strip().replace(" ", "")
    return color


def validate_graph_params(params: dict, required_keys: list) -> tuple:
    """
    Check all required keys exist and x/y arrays are same length.
    Returns (is_valid, error_message)
    """
    for key in required_keys:
        if key not in params:
            return False, f"Missing required parameter: {key}"

    if "x_values" in params and "y_values" in params:
        if len(params["x_values"]) != len(params["y_values"]):
            return False, "x_values and y_values must have same length"

    return True, ""
