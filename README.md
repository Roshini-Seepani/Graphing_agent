# Graph Agent

Graph Agent is a Streamlit application that converts natural-language prompts, uploaded graph images, and CSV data into clean plotted graphs.

## Features

- Multi-input graph generation:
  - Prompt to graph
  - Image to graph (vision model)
  - CSV to graph
- Multi-turn chat for iterative refinements
- Axis range controls from the UI
- Session-based gallery with downloadable outputs
- Tool-based architecture for reliable graph rendering

## Project Structure

```text
graph_agent/
  app.py                     # Streamlit UI
  agent.py                   # LLM orchestration and tool selection
  requirements.txt
  output/
    sessions/                # Per-session generated graphs
  tools/
    graph_from_data.py       # CSV/table plotting
    graph_from_equation.py   # Equation plotting
    graph_from_image.py      # Image-extracted plotting
  utils/
    image_helper.py
    metadata_parser.py
```

## Tech Stack

- Python
- Streamlit
- OpenAI-compatible API client (configured for Groq endpoint)
- Matplotlib
- SymPy
- SciPy
- Pandas

## Prerequisites

- Python 3.10+ recommended
- A valid Groq API key

## Setup

1. Clone the repository and enter the project directory.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Configure environment variables.

### 1) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

Create a `.env` file in `graph_agent/` with:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the App

From the `graph_agent/` directory:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## Stop the App

In the terminal where Streamlit is running, press:

```bash
Ctrl+C
```

## How It Works

1. User submits prompt and optional image/CSV in the Streamlit UI.
2. `agent.py` sends the request to the selected model with function tools.
3. The model selects one tool:
   - `graph_from_equation`
   - `graph_from_data`
   - `graph_from_image`
4. The selected Python tool generates a graph using deterministic plotting code.
5. Output image is saved under `output/` and copied to session history.

## Supported Workflows

### Prompt to Equation Graph

Example prompts:

- Plot `sin(x)` from `-2pi` to `2pi`
- Plot `x^2 - 4x + 3`
- Plot both `sin(x)` and `cos(x)` on one graph

### CSV to Graph

Upload a CSV and ask:

- Make a bar chart of sales by month
- Plot revenue as a line chart

### Image to Graph

Upload a hand-drawn or reference graph image and ask for a clean reproduction.

## Notes

- Text-only models cannot process uploaded images.
- Axis controls in the UI can override tool/model-derived axis ranges.
- Generated outputs are stored locally in `output/sessions/`.

## Troubleshooting

### Missing API key

If requests fail immediately, ensure `.env` exists and `GROQ_API_KEY` is valid.

### Dependencies fail to install

Upgrade pip and retry:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Port already in use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

## Future Improvements

- Deterministic pre-routing for obvious CSV/equation requests
- History compaction to reduce token usage
- Request caching for repeated graph generations
- Better observability for latency and token cost
