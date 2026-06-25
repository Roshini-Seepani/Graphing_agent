import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from tools.graph_from_image import graph_from_image
from tools.graph_from_equation import graph_from_equation
from tools.graph_from_data import graph_from_data

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Tool Definitions ───────────────────────────────────────────────────────────
# OpenAI-format tool dicts — Groq uses the exact same format as OpenAI.
# The actual implementations live in tools/*.py — the model just decides which one to call.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "graph_from_image",
            "description": (
                "Use this when the user uploads a handdrawn or rough graph image "
                "and wants a clean matplotlib version. "
                "Carefully read all data points and axis labels from the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type":  {"type": "string",  "description": "line, bar, or scatter"},
                    "x_values":    {"type": "array",   "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
                    "y_values":    {"type": "array",   "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
                    "x_label":     {"type": "string"},
                    "y_label":     {"type": "string"},
                    "title":       {"type": "string"},
                    "line_style":  {"type": "string",  "description": "solid, dashed, or dotted"},
                    "color":       {"type": "string",  "description": "color name or hex, e.g. red, #FF0000"},
                    "background":  {"type": "string",  "description": "background color, e.g. white, black"},
                    "font_size":   {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "font size as number, e.g. 12"},
                    "curve_hint":  {"type": "string", "description": "Curve family recognised in the image: sinusoidal, exponential, logarithmic, linear, polynomial_2, polynomial_3, power. Omit for bar/scatter/unknown."},
                    "x_min": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Exact minimum x-axis value read from the image's axis labels"},
                    "x_max": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Exact maximum x-axis value read from the image's axis labels"},
                    "y_min": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Exact minimum y-axis value read from the image's axis labels"},
                    "y_max": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Exact maximum y-axis value read from the image's axis labels"},
                },
                "required": ["chart_type", "x_values", "y_values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_from_equation",
            "description": (
                "Use this when the user gives one or more mathematical equations to plot. "
                "Supports sin, cos, tan, log, exp, polynomials, implicit equations, etc. "
                "Use the 'equations' list for multiple lines on the same graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation":   {"type": "string",  "description": "Single equation (used when only one line)"},
                    "equations":  {"type": "array", "items": {"type": "string"}, "description": "List of equations to plot on the same graph, e.g. ['sin(x)', 'cos(x)']"},
                    "colors":     {"type": "array", "items": {"type": "string"}, "description": "List of colors for each equation, one per equation"},
                    "x_start":    {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Start of x range, e.g. -3.14 or '-pi'"},
                    "x_end":      {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "End of x range, e.g. 3.14 or 'pi'"},
                    "line_style": {"type": "string"},
                    "color":      {"type": "string"},
                    "title":      {"type": "string"},
                    "x_label":    {"type": "string"},
                    "y_label":    {"type": "string"},
                    "background": {"type": "string"},
                },
                "required": ["x_start", "x_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_from_data",
            "description": "Use this when the user pastes a table or CSV data and wants it visualized.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_text":   {"type": "string",  "description": "raw CSV content as a string"},
                    "chart_type": {"type": "string",  "description": "bar, line, scatter, or pie"},
                    "x_column":   {"type": "string"},
                    "y_column":   {"type": "string"},
                    "title":      {"type": "string"},
                    "color":      {"type": "string"},
                    "background": {"type": "string"},
                },
                "required": ["csv_text", "chart_type"],
            },
        },
    },
]

TOOL_MAP = {
    "graph_from_image":    graph_from_image,
    "graph_from_equation": graph_from_equation,
    "graph_from_data":     graph_from_data,
}

SYSTEM = (
    "You are a graph generation agent. "
    "Analyze the user's prompt and image (if any), then call the most appropriate tool. "
    "Always apply these defaults unless the user says otherwise: "
    "grid=true, clean white background, font_size=12. "
    "Extract all visual parameters the user mentions: colors, fonts, line styles, background. "
    "If the user says 'dark' or 'dark background', set background to #1a1a1a. "

    "TOOL SELECTION for image input: "
    "First, identify what kind of curve is shown in the image. "
    "If the image shows a smooth continuous mathematical curve that matches a known function "
    "(sine/cosine wave, parabola y=ax²+bx+c, exponential y=ae^(bx), logarithm, straight line, etc.) "
    "→ call graph_from_equation with the recognised equation. "
    "Estimate the parameters (amplitude, frequency, period, offset, coefficients) from the axis scale. "
    "Example: a sine wave with amplitude 3, period 2π → equation='3*sin(x)', x_start=-2*pi, x_end=2*pi. "
    "This gives a perfect smooth curve. Only use graph_from_image when the data is truly discrete "
    "(bar chart, scatter with no clear pattern, complex/unknown curve shape). "

    "When you do use graph_from_image: "
    "read ALL visible data points, read tick labels on both axes and pass exact axis bounds as "
    "x_min, x_max, y_min, y_max. "
    "Set curve_hint to one of: sinusoidal, exponential, logarithmic, linear, "
    "polynomial_2, polynomial_3, power — so the tool can fit a smooth curve through the points. "
    "Do NOT let matplotlib auto-scale — use the axis range shown in the original image. "
    "Always call a tool — never reply with plain text."
)

# ── Agent Entry Point ──────────────────────────────────────────────────────────

# Vision models: can read uploaded images
VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",     # fast, good vision
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # smarter, good vision
]

# Text-only models: no image support, but great for equations/CSV/data
TEXT_MODELS = [
    "llama-3.3-70b-versatile",         # best for equations and data
    "llama-3.1-8b-instant",            # fastest, lightweight
]

# Flat list for backward compat (vision first)
GROQ_MODELS = VISION_MODELS + TEXT_MODELS

def run_agent(prompt: str, image_path: str = None, model: str = GROQ_MODELS[0],
              history: list = None, axis_limits: dict = None) -> dict:
    """
    Main agent function.
    history: list of previous API messages for multi-turn conversation.
    Returns: { "tool_called": str, "output_path": str, "message": str, "api_messages": list }
    """
    # Build the new user message
    user_content = [{"type": "text", "text": prompt}]
    if image_path:
        ext = image_path.split(".")[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    # Build full message list: system + history + new user message
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
    )

    assistant_msg = response.choices[0].message
    tool_calls = assistant_msg.tool_calls
    if tool_calls:
        tool_name = tool_calls[0].function.name
        tool_args = json.loads(tool_calls[0].function.arguments)

        # Sanitize: model sometimes returns {"type":"array","value":[...]} instead of [...]
        for key in ("x_values", "y_values", "equations", "colors"):
            if key in tool_args and isinstance(tool_args[key], dict):
                tool_args[key] = tool_args[key].get("value", [])

        # Sanitize: model sometimes passes "3π/2" or "3*pi/2" strings for numeric fields
        import math as _math
        import re as _re
        def _to_float(val):
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str):
                v = val.strip()
                # Fix implicit multiplication: "3π" → "3*pi", "3pi" → "3*pi"
                v = _re.sub(r'(\d)\s*π', r'\1*pi', v)
                v = _re.sub(r'(\d)\s*pi\b', r'\1*pi', v)
                v = v.replace("π", "pi")          # standalone π → pi
                v = v.replace("∞", "inf")         # unicode minus / inf
                v = v.replace("−", "-")
                _ns = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
                _ns["__builtins__"] = {}
                try:
                    return float(eval(v, _ns))
                except Exception:
                    return val
            return val
        # Coerce all scalar numeric fields from string → float
        for key in ("x_start", "x_end", "y_start", "y_end", "font_size",
                    "x_min", "x_max", "y_min", "y_max"):
            if key in tool_args:
                tool_args[key] = _to_float(tool_args[key])

        # Coerce array numeric fields (model may send items as strings)
        for key in ("x_values", "y_values"):
            if key in tool_args and isinstance(tool_args[key], list):
                tool_args[key] = [_to_float(v) for v in tool_args[key]]

        print(f"[Agent] Calling tool : {tool_name}")
        print(f"[Agent] Arguments    :\n{json.dumps(tool_args, indent=2)}")

        result = TOOL_MAP[tool_name](tool_args, axis_limits=axis_limits or {})

        # Build updated history to return (user msg + assistant tool call + tool result)
        new_history = list(history or [])
        new_history.append({"role": "user", "content": user_content})
        new_history.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tool_calls[0].id, "type": "function",
             "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}
        ]})
        new_history.append({"role": "tool", "tool_call_id": tool_calls[0].id,
                            "content": json.dumps(tool_args)})

        return {
            "tool_called": tool_name,
            "output_path": result.get("output_path"),
            "message":     result.get("message", "Done"),
            "history":     new_history,
        }

    return {"message": "Agent could not determine what to do. Try rephrasing.", "history": list(history or [])}
