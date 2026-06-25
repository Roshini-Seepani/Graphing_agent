import os
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from utils.metadata_parser import sanitize_color


def graph_from_data(params: dict, axis_limits: dict = None) -> dict:
    axis_limits = axis_limits or {}
    try:
        df = pd.read_csv(StringIO(params["csv_text"]))

        x_col = params.get("x_column") or df.columns[0]
        y_col = params.get("y_column") or df.columns[1]

        fig, ax = plt.subplots(figsize=(8, 5))
        bg = params.get("background", "white")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        text_color = "white" if bg in ["black", "#000000", "#1a1a1a"] else "black"

        chart_type = params.get("chart_type", "bar")
        color      = sanitize_color(params.get("color", "teal"))

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
        ax.grid(True, alpha=0.3, color=text_color)

        # Apply user-specified axis limits
        if axis_limits.get("x_min") is not None or axis_limits.get("x_max") is not None:
            ax.set_xlim(left=axis_limits.get("x_min"), right=axis_limits.get("x_max"))
        if axis_limits.get("y_min") is not None or axis_limits.get("y_max") is not None:
            ax.set_ylim(bottom=axis_limits.get("y_min"), top=axis_limits.get("y_max"))

        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        output_path = "output/data_graph.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {"output_path": output_path, "message": "Graph generated from data."}

    except Exception as e:
        return {"message": f"Data error: {str(e)}"}
