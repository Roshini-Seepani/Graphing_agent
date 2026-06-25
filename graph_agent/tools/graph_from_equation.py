import os
import math as _math
import re as _re
from fractions import Fraction
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from utils.metadata_parser import sanitize_color


def _pi_tick_label(val: float) -> str:
    """Return a π-fraction string for val, e.g. 1.5707… → 'π/2'."""
    if abs(val) < 1e-9:
        return "0"
    frac = Fraction(val / _math.pi).limit_denominator(8)
    n, d = frac.numerator, frac.denominator
    if abs(n / d - val / _math.pi) > 0.01:
        return f"{val:.3g}"  # not a clean π fraction
    sign = "-" if n < 0 else ""
    n = abs(n)
    if d == 1:
        return f"{sign}π" if n == 1 else f"{sign}{n}π"
    return f"{sign}π/{d}" if n == 1 else f"{sign}{n}π/{d}"


_TRIG_RE = _re.compile(r'\b(sin|cos|tan|csc|sec|cot|asin|acos|atan2?)\b')


def _has_trig(equations: list) -> bool:
    """Return True if any equation contains a trig function."""
    return any(_TRIG_RE.search(e.lower()) for e in equations)


def _best_pi_step(span: float) -> float:
    """Pick a π-multiple step that yields ~5-8 ticks for the given span."""
    candidates = [_math.pi / 4, _math.pi / 2, _math.pi,
                  2 * _math.pi, 4 * _math.pi]
    for step in candidates:
        n = span / step
        if 4 <= n <= 10:
            return step
    # span very large or very small — fall back to something reasonable
    return max(candidates[0], round(span / 6 / _math.pi) * _math.pi or _math.pi)


def _apply_pi_ticks(ax, x_lo: float, x_hi: float, force: bool = False) -> None:
    """Label x-axis ticks as π-fractions.

    force=True  → always apply (e.g. trig equation detected).
    force=False → only apply when both limits are already π-multiples.
    """
    def _is_pi_frac(v):
        if abs(v) < 1e-9:
            return True
        f = Fraction(v / _math.pi).limit_denominator(4)
        return abs(float(f) - v / _math.pi) < 0.02

    if not force and not (_is_pi_frac(x_lo) and _is_pi_frac(x_hi)):
        return

    span = x_hi - x_lo
    if span <= 0:
        return
    step = _best_pi_step(span)

    # Snap first tick to a clean π multiple at or after x_lo
    first = _math.floor(x_lo / step) * step
    ticks = []
    t = first
    while t <= x_hi + 1e-9:
        if t >= x_lo - 1e-9:
            snapped = round(t / step) * step   # exact multiple, no float drift
            ticks.append(snapped)
        t += step

    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([_pi_tick_label(v) for v in ticks])

# Map Unicode math symbols → sympy-compatible strings
_MATH_SUBS = {
    "π": "pi",
    "∞": "oo",
    "√": "sqrt",
    "²": "**2",
    "³": "**3",
    "×": "*",
    "÷": "/",
    "−": "-",   # Unicode minus
    "·": "*",
}

def _normalize_eq(eq: str) -> str:
    """Normalize equation: convert Unicode symbols and handle special cases."""
    for sym, rep in _MATH_SUBS.items():
        eq = eq.replace(sym, rep)
    
    # Convert e^x or e^(...) to exp(...) for proper NumPy evaluation
    import re as _re_normalize
    # First handle e^(expr) where expr is in parentheses
    eq = _re_normalize.sub(r'e\^\(([^)]+)\)', r'exp(\1)', eq)  # e^(x+1) → exp(x+1)
    # Then handle e^single_variable or e^single_number
    eq = _re_normalize.sub(r'e\^([a-zA-Z0-9_]+)', r'exp(\1)', eq)  # e^x → exp(x)
    
    # Handle common mathematical functions: exp, log, ln, sqrt
    eq = eq.replace("ln(", "log(")   # ln → log (natural log in SymPy)
    eq = eq.replace("log10(", "log(")  # log10 handling
    
    return eq

DEFAULT_COLORS = ["royalblue", "crimson", "seagreen", "darkorange", "purple", "teal"]


def _plot_one(ax, equation: str, x_start: float, x_end: float,
              color: str, linestyle: str, text_color: str):
    """Plot a single equation (explicit or implicit) onto ax."""
    equation = _normalize_eq(equation)
    x_sym = sp.Symbol("x")
    y_sym = sp.Symbol("y")

    is_implicit = "=" in equation or ("y" in equation and "x" in equation)

    if is_implicit:
        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            expr = sp.sympify(lhs_str.strip()) - sp.sympify(rhs_str.strip())
        else:
            expr = sp.sympify(equation)
        F  = sp.lambdify((x_sym, y_sym), expr, "numpy")
        xs = np.linspace(x_start, x_end, 800)
        ys = np.linspace(x_start, x_end, 800)
        X, Y = np.meshgrid(xs, ys)
        Z = F(X, Y)
        ax.contour(X, Y, Z, levels=[0], colors=[color], linewidths=2)
        ax.set_aspect("equal")
    else:
        try:
            expr   = sp.sympify(equation)
            
            # Use ['numpy', 'math'] modules for proper numeric evaluation
            # This ensures exp, log, sin, cos, etc. work with numpy arrays
            f      = sp.lambdify(x_sym, expr, ['numpy', 'math'])
            x_vals = np.linspace(x_start, x_end, 800)
            
            try:
                y_vals = f(x_vals)
                
                # Convert to numpy array and filter inf/nan
                y_vals = np.asarray(y_vals, dtype=float)
                # Clip extreme values to prevent overflow in visualization
                y_max_clip = 1e8  # Arbitrary large limit
                y_min_clip = -1e8
                y_vals = np.clip(y_vals, y_min_clip, y_max_clip, out=np.empty_like(y_vals))
                # Mark infinities and invalid values as NaN
                y_vals[np.isinf(y_vals)] = np.nan
                
                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle,
                        linewidth=2, label=equation)
            except (OverflowError, FloatingPointError, ValueError) as e:
                print(f"[Warning] Evaluation error for '{equation}': {e}")
                # Fallback: point-by-point evaluation with error catching
                try:
                    y_vals = []
                    for x_val in x_vals:
                        try:
                            y_val = float(f(x_val))
                            # Clamp extreme values
                            if abs(y_val) > 1e8:
                                y_val = np.nan
                            y_vals.append(y_val)
                        except (OverflowError, ValueError, TypeError):
                            y_vals.append(np.nan)
                    y_vals = np.array(y_vals)
                    ax.plot(x_vals, y_vals, color=color, linestyle=linestyle,
                            linewidth=2, label=equation)
                except Exception as e2:
                    print(f"[Error] Fallback also failed for '{equation}': {e2}")
                    
        except Exception as e:
            print(f"[Error] Could not parse equation '{equation}': {e}")


def graph_from_equation(params: dict, axis_limits: dict = None) -> dict:
    axis_limits = axis_limits or {}
    try:
        # Support both single 'equation' and multiple 'equations'
        equations = params.get("equations") or []
        if not equations and params.get("equation"):
            equations = [params["equation"]]
        if not equations:
            return {"message": "No equation provided."}

        x_start   = params["x_start"]
        x_end     = params["x_end"]
        bg        = params.get("background", "white")
        linestyle = params.get("line_style", "solid")
        title     = params.get("title", " & ".join(equations))
        colors    = params.get("colors") or []
        # Sanitize all colors
        colors    = [sanitize_color(c) for c in colors]
        # Fill missing colors from defaults
        while len(colors) < len(equations):
            colors.append(DEFAULT_COLORS[len(colors) % len(DEFAULT_COLORS)])
        # Single-equation color override
        if len(equations) == 1 and params.get("color"):
            colors[0] = sanitize_color(params["color"])

        text_color = "white" if bg in ["black", "#000000", "#1a1a1a"] else "black"

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        for eq, col in zip(equations, colors):
            _plot_one(ax, eq, x_start, x_end, col, linestyle, text_color)

        ax.set_xlabel(params.get("x_label", "x"), color=text_color)
        ax.set_ylabel(params.get("y_label", "y"), color=text_color)
        ax.set_title(title, color=text_color)
        ax.tick_params(colors=text_color)
        ax.axhline(0, color=text_color, linewidth=0.5)
        ax.axvline(0, color=text_color, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
        ax.grid(True, alpha=0.3, color=text_color)

        if len(equations) > 1:
            legend = ax.legend(facecolor=bg, edgecolor=text_color,
                               labelcolor=text_color, fontsize=10)

        # Apply user-specified axis limits (always override model output)
        if axis_limits.get("x_min") is not None or axis_limits.get("x_max") is not None:
            ax.set_xlim(
                left=axis_limits.get("x_min"),
                right=axis_limits.get("x_max"),
            )
        if axis_limits.get("y_min") is not None or axis_limits.get("y_max") is not None:
            ax.set_ylim(
                bottom=axis_limits.get("y_min"),
                top=axis_limits.get("y_max"),
            )

        # Apply π-fraction x-axis tick labels:
        # force=True whenever any equation contains sin/cos/tan/etc.
        lo, hi = ax.get_xlim()
        norm_eqs = [_normalize_eq(e) for e in equations]
        _apply_pi_ticks(ax, lo, hi, force=_has_trig(norm_eqs))

        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        output_path = "output/equation_graph.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {"output_path": output_path, "message": f"Plotted: {', '.join(equations)}"}

    except Exception as e:
        return {"message": f"Equation error: {str(e)}"}

