import os
import numpy as np
import matplotlib.pyplot as plt
from utils.metadata_parser import validate_graph_params, sanitize_color


# ── Curve fitting helpers ──────────────────────────────────────────────────────

def _fit_curve(x: np.ndarray, y: np.ndarray, hint: str, x_smooth: np.ndarray):
    """
    Fit x/y data to the curve family named by hint.
    Returns y_smooth (smooth fitted values over x_smooth), or None on failure.
    """
    try:
        from scipy import optimize as sco

        if hint == "linear":
            c = np.polyfit(x, y, 1)
            return np.polyval(c, x_smooth)

        if hint == "polynomial_2":
            c = np.polyfit(x, y, 2)
            return np.polyval(c, x_smooth)

        if hint == "polynomial_3":
            c = np.polyfit(x, y, 3)
            return np.polyval(c, x_smooth)

        if hint == "sinusoidal":
            A0 = (y.max() - y.min()) / 2 or 1.0
            D0 = (y.max() + y.min()) / 2
            span = float(x.max() - x.min()) or 1.0
            B0 = 2 * np.pi / span
            def _sin(xv, A, B, C, D): return A * np.sin(B * xv + C) + D
            popt, _ = sco.curve_fit(_sin, x, y, p0=[A0, B0, 0.0, D0], maxfev=20000)
            return _sin(x_smooth, *popt)

        if hint == "exponential":
            def _exp(xv, a, b, c): return a * np.exp(b * xv) + c
            p0 = [1.0, 0.1, float(y.mean())]
            popt, _ = sco.curve_fit(_exp, x, y, p0=p0, maxfev=20000)
            return _exp(x_smooth, *popt)

        if hint == "logarithmic":
            def _log(xv, a, b): return a * np.log(np.abs(xv) + 1e-9) + b
            popt, _ = sco.curve_fit(_log, x, y, maxfev=10000)
            return _log(x_smooth, *popt)

        if hint == "power":
            def _pow(xv, a, b): return a * np.power(np.abs(xv) + 1e-9, b)
            popt, _ = sco.curve_fit(_pow, x, y, p0=[1.0, 1.0], maxfev=10000)
            return _pow(x_smooth, *popt)

    except Exception:
        pass
    return None


def graph_from_image(params: dict, axis_limits: dict = None) -> dict:
    axis_limits = axis_limits or {}
    valid, error = validate_graph_params(params, ["chart_type", "x_values", "y_values"])
    if not valid:
        return {"message": f"Validation error: {error}"}

    fig, ax = plt.subplots(figsize=(8, 5))

    # Background
    bg = params.get("background", "white")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    text_color = "white" if bg in ["black", "#000000", "#1a1a1a"] else "black"

    # Plot
    chart_type = params.get("chart_type", "line")
    color      = sanitize_color(params.get("color", "steelblue"))
    linestyle  = params.get("line_style", "solid")
    curve_hint = params.get("curve_hint", "").strip().lower()

    x_vals = np.array(params["x_values"], dtype=float)
    y_vals = np.array(params["y_values"], dtype=float)

    if chart_type == "line":
        # Determine smooth x range from image-extracted bounds or data range
        x_lo = params.get("x_min") if params.get("x_min") is not None else float(x_vals.min())
        x_hi = params.get("x_max") if params.get("x_max") is not None else float(x_vals.max())
        x_smooth = np.linspace(x_lo, x_hi, 800)

        fitted = None
        if curve_hint and curve_hint != "discrete" and len(x_vals) >= 3:
            fitted = _fit_curve(x_vals, y_vals, curve_hint, x_smooth)

        if fitted is not None:
            # Smooth fitted curve + original extracted points as reference markers
            ax.plot(x_smooth, fitted, color=color, linestyle=linestyle, linewidth=2)
            ax.scatter(x_vals, y_vals, color=color, s=25, zorder=5, alpha=0.5, label="extracted points")
        else:
            # Fallback: connect sparse extracted points directly
            ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2, marker="o", markersize=4)
    elif chart_type == "bar":
        ax.bar(x_vals, y_vals, color=color)
    elif chart_type == "scatter":
        ax.scatter(x_vals, y_vals, color=color)

    # Labels
    font_size = params.get("font_size", 11)
    ax.set_xlabel(params.get("x_label", ""), color=text_color, fontsize=font_size)
    ax.set_ylabel(params.get("y_label", ""), color=text_color, fontsize=font_size)
    ax.set_title(params.get("title", ""),    color=text_color, fontsize=font_size + 2)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    ax.grid(True, alpha=0.3, color=text_color)

    # Layer 1: apply axis bounds extracted from the image by the model (exact match)
    img_x_min = params.get("x_min")
    img_x_max = params.get("x_max")
    img_y_min = params.get("y_min")
    img_y_max = params.get("y_max")
    if img_x_min is not None or img_x_max is not None:
        ax.set_xlim(left=img_x_min, right=img_x_max)
    if img_y_min is not None or img_y_max is not None:
        ax.set_ylim(bottom=img_y_min, top=img_y_max)

    # Layer 2: user slider overrides (only if explicitly set in the UI)
    if axis_limits.get("x_min") is not None or axis_limits.get("x_max") is not None:
        ax.set_xlim(left=axis_limits.get("x_min"), right=axis_limits.get("x_max"))
    if axis_limits.get("y_min") is not None or axis_limits.get("y_max") is not None:
        ax.set_ylim(bottom=axis_limits.get("y_min"), top=axis_limits.get("y_max"))

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    output_path = "output/graph_from_image.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"output_path": output_path, "message": "Graph generated from image."}
