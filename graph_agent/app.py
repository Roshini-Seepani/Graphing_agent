import streamlit as st
import tempfile
import os
import shutil
import uuid
from datetime import datetime
from agent import run_agent, GROQ_MODELS, VISION_MODELS, TEXT_MODELS

st.set_page_config(page_title="Graph Agent", layout="wide")
st.title("Graph Agent")
st.caption("Describe any graph. Upload a sketch or paste data. Agent does the rest.")

# ── Session state init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

SESSION_DIR = os.path.join("output", "sessions", st.session_state.session_id)
os.makedirs(SESSION_DIR, exist_ok=True)

# ── Sidebar — Gallery only ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Gallery")
    if st.button("New conversation"):
        st.session_state.history    = []
        st.session_state.chat       = []
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        st.rerun()
    st.divider()
    sessions_root = os.path.join("output", "sessions")
    if os.path.exists(sessions_root):
        all_sessions = sorted(os.listdir(sessions_root), reverse=True)
        for sess in all_sessions:
            sess_path = os.path.join(sessions_root, sess)
            images = sorted([f for f in os.listdir(sess_path) if f.endswith(".png")])
            if not images:
                continue
            label = "Current" if sess == st.session_state.session_id else sess
            with st.expander(f"{'🟢 ' if sess == st.session_state.session_id else ''}{label} ({len(images)} graphs)"):
                for img_file in images:
                    img_path = os.path.join(sess_path, img_file)
                    st.image(img_path, use_container_width=True)
                    with open(img_path, "rb") as f:
                        st.download_button(
                            "Download", f, file_name=img_file,
                            key=f"gal_{sess}_{img_file}",
                        )
    else:
        st.caption("No graphs yet.")

# ── Top bar — model + uploads in one row ──────────────────────────────────────
col_model, col_img, col_csv = st.columns([2, 2, 2])
with col_model:
    # Build labeled options: group vision and text-only
    model_options  = (["── Vision models ──"] + VISION_MODELS +
                      ["── Text-only models ──"] + TEXT_MODELS)
    model_defaults = [m for m in model_options if not m.startswith("──")]
    raw_sel = st.selectbox("Model", model_options, index=1,
                           label_visibility="collapsed")
    # If user picks a separator line, fall back to first vision model
    model = raw_sel if not raw_sel.startswith("──") else VISION_MODELS[0]
    if model in TEXT_MODELS and not raw_sel.startswith("──"):
        st.caption("⚠️ Text-only — image upload ignored")
with col_img:
    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
with col_csv:
    csv_file = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

# ── Axis range row ─────────────────────────────────────────────────────────────
with st.expander("Set axis ranges (optional)", expanded=False):
    rc1, rc2 = st.columns(2)
    with rc1:
        x_auto = st.checkbox("X auto", value=False)
        x_range = st.slider("X range", min_value=-100.0, max_value=100.0,
                            value=(-5.0, 10.0), step=0.5, format="%g",
                            disabled=x_auto)
        x_min, x_max = (None, None) if x_auto else x_range
    with rc2:
        y_auto = st.checkbox("Y auto", value=False)
        y_range = st.slider("Y range", min_value=-100.0, max_value=100.0,
                            value=(-3.0, 10.0), step=0.5, format="%g",
                            disabled=y_auto)
        y_min, y_max = (None, None) if y_auto else y_range

# ── Chat area ──────────────────────────────────────────────────────────────────
for i, entry in enumerate(st.session_state.chat):
    with st.chat_message(entry["role"]):
        st.write(entry["text"])
        if entry.get("image"):
            st.image(entry["image"], use_container_width=True)
            with open(entry["image"], "rb") as f:
                st.download_button("Download", f, file_name="graph.png",
                                   key=f"dl_{i}")

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input("Describe a graph, or refine the last one...")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat.append({"role": "user", "text": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            image_path = None
            if uploaded and model in VISION_MODELS:
                suffix = "." + uploaded.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    image_path = tmp.name
            elif uploaded and model in TEXT_MODELS:
                st.warning("Image ignored — selected model cannot process images. Switch to a Vision model.")

            # If CSV uploaded, inject its content into the prompt
            final_prompt = prompt
            if csv_file:
                csv_text = csv_file.read().decode("utf-8")
                final_prompt = f"{prompt}\n\nCSV data:\n{csv_text}"

            # Append axis range hints so the model passes correct x_start/x_end
            range_hints = []
            if x_min is not None: range_hints.append(f"x min = {x_min}")
            if x_max is not None: range_hints.append(f"x max = {x_max}")
            if y_min is not None: range_hints.append(f"y min = {y_min}")
            if y_max is not None: range_hints.append(f"y max = {y_max}")
            if range_hints:
                final_prompt += "\n\nAxis ranges: " + ", ".join(range_hints)

            result = run_agent(
                final_prompt,
                image_path=image_path,
                model=model,
                history=st.session_state.history,
                axis_limits={
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                },
            )

            if image_path and os.path.exists(image_path):
                os.unlink(image_path)

        st.session_state.history = result.get("history", st.session_state.history)

        if result.get("output_path"):
            # Copy to session folder with a unique timestamped filename
            idx       = len([e for e in st.session_state.chat if e.get("image")])
            tool_name = result.get("tool_called", "graph")
            dest_name = f"{idx:03d}_{tool_name}.png"
            dest_path = os.path.join(SESSION_DIR, dest_name)
            shutil.copy2(result["output_path"], dest_path)

            st.write(result.get("message", "Done"))
            st.image(dest_path, use_container_width=True)
            with open(dest_path, "rb") as f:
                st.download_button("Download Graph", f, file_name=dest_name)
            st.session_state.chat.append({
                "role":  "assistant",
                "text":  result.get("message", "Done"),
                "image": dest_path,
            })
        else:
            msg = result.get("message", "No output generated.")
            st.info(msg)
            st.session_state.chat.append({"role": "assistant", "text": msg})


