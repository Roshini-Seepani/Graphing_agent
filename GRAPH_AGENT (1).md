# Graph Agent — Implementation Guide

A multimodal AI agent that converts handdrawn sketches, equations, and datasets into
publication-quality matplotlib graphs using Gemini Vision + function calling.

---

## Project Structure

```
graph_agent/
│
├── agent.py                    # Core agent — Gemini + function calling loop
├── tools/
│   ├── __init__.py
│   ├── graph_from_image.py     # Tool 1 — handdrawn image → matplotlib graph
│   ├── graph_from_equation.py  # Tool 2 — equation string → matplotlib graph
│   └── graph_from_data.py      # Tool 3 — CSV/table → matplotlib graph
│
├── utils/
│   ├── __init__.py
│   ├── image_helper.py         # image reading, base64 encoding for Gemini
│   └── metadata_parser.py      # validates + cleans agent JSON output
│
├── output/                     # all generated graphs saved here
├── app.py                      # Streamlit UI
├── requirements.txt
└── .env                        # API keys (never commit this)
```

---

## Setup

### 1. Install dependencies

```bash
pip install google-generativeai matplotlib sympy pandas streamlit python-dotenv pillow
```

### 2. Get Gemini API key

- Go to https://aistudio.google.com
- Create API key (free, no credit card)
- Add to `.env`:

```
GEMINI_API_KEY=your_key_here
```

---

## File by File Implementation

---

### `agent.py` — The Brain

This is the most important file. The agent:
1. Receives user prompt + optional image
2. Sends to Gemini with tool definitions
3. Gemini returns which tool to call + arguments
4. Agent calls the right tool
5. Returns output path

```python
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from utils.image_helper import load_image_as_base64
from tools.graph_from_image import graph_from_image
from tools.graph_from_equation import graph_from_equation
from tools.graph_from_data import graph_from_data

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Tool Definitions (what Gemini can call) ---

TOOLS = [
    {
        "name": "graph_from_image",
        "description": (
            "Use this when the user uploads a handdrawn or rough graph image "
            "and wants a clean matplotlib version of it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type":  {"type": "string",  "description": "line, bar, scatter"},
                "x_values":    {"type": "array",   "items": {"type": "number"}},
                "y_values":    {"type": "array",   "items": {"type": "number"}},
                "x_label":     {"type": "string"},
                "y_label":     {"type": "string"},
                "title":       {"type": "string"},
                "line_style":  {"type": "string",  "description": "solid, dashed, dotted"},
                "color":       {"type": "string",  "description": "e.g. red, #FF0000"},
                "background":  {"type": "string",  "description": "e.g. white, black, #1a1a1a"},
                "font_size":   {"type": "number"}
            },
            "required": ["chart_type", "x_values", "y_values"]
        }
    },
    {
        "name": "graph_from_equation",
        "description": (
            "Use this when the user gives a mathematical equation to plot. "
            "Supports sin, cos, log, exp, polynomials etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equation":   {"type": "string",  "description": "e.g. sin(x)/x, x**2 + 3*x"},
                "x_start":    {"type": "number"},
                "x_end":      {"type": "number"},
                "line_style": {"type": "string"},
                "color":      {"type": "string"},
                "title":      {"type": "string"},
                "x_label":    {"type": "string"},
                "y_label":    {"type": "string"},
                "background": {"type": "string"}
            },
            "required": ["equation", "x_start", "x_end"]
        }
    },
    {
        "name": "graph_from_data",
        "description": (
            "Use this when the user pastes a table or CSV data "
            "and wants it visualized."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "csv_text":    {"type": "string", "description": "raw CSV content as string"},
                "chart_type":  {"type": "string", "description": "bar, line, scatter, pie"},
                "x_column":    {"type": "string"},
                "y_column":    {"type": "string"},
                "title":       {"type": "string"},
                "color":       {"type": "string"},
                "background":  {"type": "string"}
            },
            "required": ["csv_text", "chart_type"]
        }
    }
]

TOOL_MAP = {
    "graph_from_image":    graph_from_image,
    "graph_from_equation": graph_from_equation,
    "graph_from_data":     graph_from_data
}

# --- Agent Entry Point ---

def run_agent(prompt: str, image_path: str = None) -> dict:
    """
    Main agent function.
    Returns dict: { "tool_called": str, "output_path": str, "message": str }
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build message content
    content = [prompt]
    if image_path:
        b64, mime = load_image_as_base64(image_path)
        content.append({"mime_type": mime, "data": b64})

    # System instruction
    system = (
        "You are a graph generation agent. "
        "Analyze the user's prompt and image (if any), then call the most appropriate tool. "
        "Extract all visual parameters the user mentions (colors, fonts, line styles, background). "
        "For handdrawn graphs, carefully read all data points and axis labels from the image. "
        "Always return tool call with complete parameters."
    )

    response = model.generate_content(
        content,
        tools=TOOLS,
        system_instruction=system
    )

    # Extract tool call from response
    for part in response.candidates[0].content.parts:
        if hasattr(part, "function_call"):
            tool_name = part.function_call.name
            tool_args = dict(part.function_call.args)

            print(f"[Agent] Calling tool: {tool_name}")
            print(f"[Agent] Arguments: {json.dumps(tool_args, indent=2)}")

            result = TOOL_MAP[tool_name](tool_args)
            return {
                "tool_called": tool_name,
                "output_path": result.get("output_path"),
                "message":     result.get("message", "Done")
            }

    return {"message": "Agent could not determine what to do. Try rephrasing."}
```

---

### `utils/image_helper.py`

```python
import base64

SUPPORTED = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp"
}

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Returns (base64_string, mime_type)"""
    ext = "." + image_path.split(".")[-1].lower()
    mime = SUPPORTED.get(ext, "image/png")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, mime
```

---

### `utils/metadata_parser.py`

```python
def validate_graph_params(params: dict, required_keys: list) -> tuple[bool, str]:
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
```

---

### `tools/graph_from_image.py` — Tool 1

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils.metadata_parser import validate_graph_params

def graph_from_image(params: dict) -> dict:
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
    color      = params.get("color", "steelblue")
    linestyle  = params.get("line_style", "solid")

    if chart_type == "line":
        ax.plot(params["x_values"], params["y_values"],
                color=color, linestyle=linestyle, linewidth=2)

    elif chart_type == "bar":
        ax.bar(params["x_values"], params["y_values"], color=color)

    elif chart_type == "scatter":
        ax.scatter(params["x_values"], params["y_values"], color=color)

    # Labels
    font_size = params.get("font_size", 11)
    ax.set_xlabel(params.get("x_label", ""), color=text_color, fontsize=font_size)
    ax.set_ylabel(params.get("y_label", ""), color=text_color, fontsize=font_size)
    ax.set_title(params.get("title", ""),    color=text_color, fontsize=font_size + 2)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    plt.tight_layout()
    output_path = "output/graph_from_image.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"output_path": output_path, "message": "Graph generated from image."}
```

---

### `tools/graph_from_equation.py` — Tool 2

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def graph_from_equation(params: dict) -> dict:
    try:
        x = sp.Symbol("x")
        expr = sp.sympify(params["equation"])     # safely parses user string
        f    = sp.lambdify(x, expr, "numpy")

        x_vals = np.linspace(params["x_start"], params["x_end"], 800)
        y_vals = f(x_vals)

        fig, ax = plt.subplots(figsize=(8, 5))
        bg = params.get("background", "white")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        text_color = "white" if bg in ["black", "#000000", "#1a1a1a"] else "black"

        ax.plot(x_vals, y_vals,
                color=params.get("color", "royalblue"),
                linestyle=params.get("line_style", "solid"),
                linewidth=2)

        ax.set_xlabel(params.get("x_label", "x"), color=text_color)
        ax.set_ylabel(params.get("y_label", "y"), color=text_color)
        ax.set_title(params.get("title", params["equation"]), color=text_color)
        ax.tick_params(colors=text_color)
        ax.axhline(0, color=text_color, linewidth=0.5)
        ax.axvline(0, color=text_color, linewidth=0.5)

        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)

        plt.tight_layout()
        output_path = "output/equation_graph.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {"output_path": output_path, "message": f"Plotted: {params['equation']}"}

    except Exception as e:
        return {"message": f"Equation error: {str(e)}"}
```

---

### `tools/graph_from_data.py` — Tool 3

> **Note on seaborn:** Stick to matplotlib for all chart types here. Seaborn's opinionated
> theming conflicts with agent-supplied style params (background, colors). The one exception:
> if you add statistical chart types (heatmap, violin, boxplot), use seaborn only for those
> since doing them in raw matplotlib is unnecessarily painful. Don't mix otherwise.

```python
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

def graph_from_data(params: dict) -> dict:
    try:
        df = pd.read_csv(StringIO(params["csv_text"]))

        x_col = params.get("x_column", df.columns[0])
        y_col = params.get("y_column", df.columns[1])

        fig, ax = plt.subplots(figsize=(8, 5))
        bg = params.get("background", "white")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        text_color = "white" if bg in ["black", "#000000", "#1a1a1a"] else "black"

        chart_type = params.get("chart_type", "bar")
        color      = params.get("color", "teal")

        if chart_type == "bar":
            ax.bar(df[x_col], df[y_col], color=color)
        elif chart_type == "line":
            ax.plot(df[x_col], df[y_col], color=color, linewidth=2)
        elif chart_type == "scatter":
            ax.scatter(df[x_col], df[y_col], color=color)
        elif chart_type == "pie":
            ax.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%")

        ax.set_xlabel(x_col, color=text_color)
        ax.set_ylabel(y_col, color=text_color)
        ax.set_title(params.get("title", f"{y_col} vs {x_col}"), color=text_color)
        ax.tick_params(colors=text_color)

        plt.tight_layout()
        output_path = "output/data_graph.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {"output_path": output_path, "message": "Graph generated from data."}

    except Exception as e:
        return {"message": f"Data error: {str(e)}"}
```

---

### `app.py` — Streamlit UI

```python
import streamlit as st
import tempfile
import os
from agent import run_agent

st.set_page_config(page_title="Graph Agent", layout="centered")
st.title("Graph Agent")
st.caption("Describe any graph. Upload a sketch or paste data. Agent does the rest.")

# Input
uploaded = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg"])
prompt   = st.text_area("What do you want?",
                         placeholder="e.g. plot sin(x)/x from -4π to 4π, black background, red dotted line")

if st.button("Generate", type="primary"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Agent thinking..."):
            image_path = None
            if uploaded:
                # Save uploaded file temporarily
                suffix = "." + uploaded.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    image_path = tmp.name

            result = run_agent(prompt, image_path)

            # Cleanup temp file
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)

        if result.get("output_path"):
            st.success(result.get("message", "Done"))
            st.image(result["output_path"], use_column_width=True)
            with open(result["output_path"], "rb") as f:
                st.download_button("Download Graph", f, file_name="graph.png")
        else:
            st.info(result.get("message", "No output generated."))
```

---

## How to Run

```bash
# 1. Clone / create the project folder
cd graph_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key to .env

# 4. Run the app
streamlit run app.py
```

---

## Example Prompts to Test

| Prompt | Tool Called |
|---|---|
| `plot sin(x)/x from -4π to 4π, red dotted, black background` | graph_from_equation |
| `[upload handdrawn graph] make this a clean line graph, blue, white background` | graph_from_image |
| `[paste CSV] show me a bar chart of sales vs month, teal bars` | graph_from_data |
| `plot x^2 - 4x + 4, green solid line, dark background` | graph_from_equation |

---

## What Each Team Member Should Own

| File | Concepts Covered |
|---|---|
| `agent.py` | Function calling, Gemini API, JSON parsing, decision logic |
| `graph_from_image.py` | matplotlib, chart types, vision parameter extraction |
| `graph_from_equation.py` | sympy, numpy, math → code |
| `graph_from_data.py` | pandas, CSV parsing, chart type selection logic |
| `app.py` | Streamlit, file upload handling, UX |

---

## Interview Questions to Prepare

1. Why matplotlib instead of an image generation API?
2. How does Gemini decide which tool to call?
3. What happens if Gemini extracts wrong data points from the image?
4. How would you add a new tool to this agent?
5. What is function calling and how is it different from a normal API call?

---

## Possible Addons (After Core is Done)

- **Email crafter** — agent drafts a professional email body describing the graph (LLM only, no sending)
- **Telegram sender** — send generated graph to a Telegram chat via Bot API (free, reliable)
- Multi-graph layout: *"show both sin(x) and cos(x) on same plot"*
- Session history: remember previous graphs in conversation
- Export to PDF report with all generated graphs
- Voice input: speak the graph description
