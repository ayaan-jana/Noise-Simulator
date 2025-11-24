import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import time as pytime
import matplotlib.pyplot as plt
from io import BytesIO
import re
import psycopg2
import toml
from contextlib import contextmanager
from pathlib import Path

# -------------------------
# Manual secrets path (update if needed)
# -------------------------
CUSTOM_SECRET_PATH = r"C:\Users\ayanj\Desktop\Simulator\secrets.toml"
APP_DIR = Path(__file__).resolve().parent
CSS_FILE_PATH = APP_DIR / "styles.css"


def inject_custom_css():
    try:
        css_content = CSS_FILE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning(f"Custom CSS file not found at {CSS_FILE_PATH}.")
        return
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

try:
    my_secrets = toml.load(CUSTOM_SECRET_PATH)
except Exception as e:
    st.error(f"Failed to load secrets.toml from {CUSTOM_SECRET_PATH}: {e}")
    my_secrets = {}

# Helper to check DB config keys
def db_config_valid(secrets: dict):
    required = ["db_host", "db_port", "db_user", "db_password", "db_name"]
    for k in required:
        if k not in secrets or secrets.get(k) in (None, ""):
            return False, k
    return True, None

# ==========================================================
# POSTGRESQL SUPPORT
# ==========================================================
def get_db_connection():
    ok, missing = db_config_valid(my_secrets)
    if not ok:
        raise RuntimeError(f"Missing DB secret: {missing}. Check your secrets TOML.")
    port_val = my_secrets.get("db_port")
    try:
        port_val = int(port_val)
    except Exception:
        raise RuntimeError(f"Invalid db_port in secrets: {my_secrets.get('db_port')!r}")
    return psycopg2.connect(
        host=my_secrets.get("db_host"),
        port=port_val,
        user=my_secrets.get("db_user"),
        password=my_secrets.get("db_password"),
        database=my_secrets.get("db_name"),
    )

def create_table_if_not_exists():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS noise_stream (
            id SERIAL PRIMARY KEY,
            time_value DOUBLE PRECISION,
            clean_y DOUBLE PRECISION,
            noisy_y DOUBLE PRECISION,
            noise_value DOUBLE PRECISION,
            clean_x DOUBLE PRECISION,
            noisy_x DOUBLE PRECISION,
            noise_value_x DOUBLE PRECISION,
            noise_value_y DOUBLE PRECISION,
            ts TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_record_pg(record: dict):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO noise_stream
        (time_value, clean_y, noisy_y, noise_value, clean_x, noisy_x, noise_value_x, noise_value_y)
        VALUES (%(time_value)s, %(clean_y)s, %(noisy_y)s, %(noise_value)s,
                %(clean_x)s, %(noisy_x)s, %(noise_value_x)s, %(noise_value_y)s);
    """, record)
    conn.commit()
    cur.close()
    conn.close()

# ==========================================================
# SAFE EXPRESSION EVALUATOR
# ==========================================================
_ALLOWED_NAMES = {
    "t", "x", "np", "sin", "cos", "tan", "exp", "sqrt", "log",
    "abs", "pi", "e"
}
def safe_expr_check(expr: str):
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")
    if not re.match(r'^[0-9\.\+\-\*\/\%\(\)\,\s\w]*$', expr):
        raise ValueError("Expression contains invalid/special characters.")
    names = re.findall(r'[A-Za-z_]\w*', expr)
    for name in names:
        if name not in _ALLOWED_NAMES:
            raise ValueError(f"Name '{name}' is not allowed in expression.")
    return True

def eval_expr(x, expr):
    safe_expr_check(expr)
    allowed = {
        't': x, 'x': x, 'np': np,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'sqrt': np.sqrt, 'log': np.log,
        'abs': np.abs, 'pi': np.pi, 'e': np.e
    }
    return eval(expr, {"__builtins__": {}}, allowed)

# ==========================================================
# MOCK DB SEND
# ==========================================================
def mock_send_to_db(record):
    print("STREAM:", record)

# ==========================================================
# PLOTTING FUNCTION (returns CSV and PNG bytes too)
# ==========================================================
def plot_to_streamlit(
    time_values, y, noisy=None, title="Plot",
    download_key=None, download_filename="timeseries.csv",
    png_key=None, clean_color="#1f77b4", noisy_color="#ff7f0e",
    clean_linewidth=1.0, noisy_linewidth=1.5, time_deltas=None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_values, y, label="Clean", linewidth=clean_linewidth, color=clean_color)
    if noisy is not None:
        ax.plot(time_values, noisy, label="Noisy", linewidth=noisy_linewidth, color=noisy_color)
    ax.legend(); ax.set_title(title); ax.grid(True)
    st.pyplot(fig)

    # CSV bytes
    csv_buf = BytesIO()
    data = {"time": time_values, "clean_y": y}
    if time_deltas is not None:
        data["time_delta"] = time_deltas
    if noisy is not None:
        data["noisy_y"] = noisy
        data["noise_value"] = np.asarray(noisy) - np.asarray(y)
    df = pd.DataFrame(data)
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    # PNG bytes
    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
    png_buf.seek(0)

    # Stats
    stats = None
    if noisy is not None:
        noise = np.asarray(noisy) - np.asarray(y)
        stats = {
            "mean": float(np.mean(noise)),
            "variance": float(np.var(noise)),
            "max_diff": float(np.max(np.abs(noise))),
        }
        st.caption(f"Mean noise: {stats['mean']:.4f} · Var: {stats['variance']:.4f} · Max diff: {stats['max_diff']:.4f}")

    return df, stats, csv_buf.getvalue(), png_buf.getvalue()

# ==========================================================
# NOISE (NUMPY)
# ==========================================================
def gaussian_noise(size, sigma):
    return np.random.normal(0.0, sigma, size)
def uniform_noise(size, amp):
    return np.random.uniform(-amp, amp, size)

# ==========================================================
# Misc helpers
# ==========================================================
def ordinal(n: int) -> str:
    # returns "1st", "2nd", "3rd", "4th", ...
    if 10 <= (n % 100) <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

DEFAULT_TIME_CONFIG = {
    "time_unit": "Milliseconds",
    "time_start": 0.0,
    "time_end": 20000.0,
    "time_interval": 50.0,
}


def ensure_time_defaults(target: dict):
    for k, v in DEFAULT_TIME_CONFIG.items():
        target.setdefault(k, v)


def time_unit_multiplier(unit: str) -> float:
    return 0.001 if unit == "Milliseconds" else 1.0


def validate_time_config(cfg: dict, label: str):
    start = float(cfg.get("time_start", 0.0))
    end = float(cfg.get("time_end", 0.0))
    interval = float(cfg.get("time_interval", 0.0))
    if end <= start:
        st.error(f"{label}: End must be greater than Start.")
        st.stop()
    if interval <= 0:
        st.error(f"{label}: Interval must be greater than 0.")
        st.stop()


def build_time_values(cfg: dict):
    start = float(cfg.get("time_start", 0.0))
    end = float(cfg.get("time_end", 0.0))
    interval = float(cfg.get("time_interval", 1.0))
    values = np.arange(start, end, interval)
    if values.size == 0 or values[-1] < end:
        values = np.append(values, end)
    t_vals = values
    return t_vals


def generate_time_series(cfg: dict, offset_seconds: float, label: str):
    ensure_time_defaults(cfg)
    validate_time_config(cfg, label)
    t_eval = build_time_values(cfg)
    mult = time_unit_multiplier(cfg.get("time_unit", "Milliseconds"))
    local_seconds = (t_eval - t_eval[0]) * mult
    global_times = offset_seconds + local_seconds
    return t_eval, global_times, float(global_times[-1])

inject_custom_css()

@contextmanager
def sidebar_card(title: str):
    st.sidebar.markdown(
        f"<div class='sidebar-card'><div class='sidebar-card-title'>{title}</div>",
        unsafe_allow_html=True,
    )
    yield
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

def render_download_button(label, data, filename, mime, key):
    st.markdown("<div class='download-card'>", unsafe_allow_html=True)
    st.download_button(label, data, file_name=filename, mime=mime, key=key)
    st.markdown("</div>", unsafe_allow_html=True)


def render_time_inputs_ui(item: dict, prefix: str):
    unit_options = ["Milliseconds", "Seconds"]
    cols = st.columns([0.2, 0.266, 0.266, 0.268])
    cols[0].markdown("<div class='expr-label small'>Time Unit</div>", unsafe_allow_html=True)
    cols[1].markdown("<div class='expr-label small'>Start</div>", unsafe_allow_html=True)
    cols[2].markdown("<div class='expr-label small'>End</div>", unsafe_allow_html=True)
    cols[3].markdown("<div class='expr-label small'>Interval</div>", unsafe_allow_html=True)
    current_unit = item.get("time_unit", "Milliseconds")
    if current_unit not in unit_options:
        current_unit = "Milliseconds"
    unit_idx = unit_options.index(current_unit)
    new_unit = cols[0].selectbox(
        f"time_unit_{prefix}",
        options=unit_options,
        index=unit_idx,
        key=f"time_unit_{prefix}",
        label_visibility="collapsed",
    )
    new_start = cols[1].number_input(
        f"time_start_{prefix}",
        value=float(item.get("time_start", 0.0)),
        step=1.0,
        key=f"time_start_{prefix}",
        label_visibility="collapsed",
    )
    new_end = cols[2].number_input(
        f"time_end_{prefix}",
        value=float(item.get("time_end", 1000.0)),
        step=1.0,
        key=f"time_end_{prefix}",
        label_visibility="collapsed",
    )
    new_interval = cols[3].number_input(
        f"time_interval_{prefix}",
        value=float(item.get("time_interval", 10.0)),
        min_value=0.0001,
        step=1.0,
        key=f"time_interval_{prefix}",
        label_visibility="collapsed",
    )
    item["time_unit"] = new_unit
    item["time_start"] = new_start
    item["time_end"] = new_end
    item["time_interval"] = new_interval

# ==========================================================
# APP UI + State Setup
# ==========================================================
st.title("Noise Simulator — Multi-Expression")
st.write("Multi-expression sequential simulation. Use 'Add' buttons to add segments/expressions.")

if "expr_list" not in st.session_state:
    base_expr = {"expr": "t * sin(0.3*t) + 3"}
    ensure_time_defaults(base_expr)
    st.session_state.expr_list = [base_expr]
else:
    for item in st.session_state.expr_list:
        ensure_time_defaults(item)

if "expr2d_list" not in st.session_state:
    base_2d = {"x": "10*sin(0.4*t)", "y": "10*cos(0.4*t)"}
    ensure_time_defaults(base_2d)
    st.session_state.expr2d_list = [base_2d]
else:
    for item in st.session_state.expr2d_list:
        ensure_time_defaults(item)

if "segments" not in st.session_state:
    base_seg = {"ax": 0.0, "ay": 0.0, "bx": 10.0, "by": 5.0}
    ensure_time_defaults(base_seg)
    st.session_state.segments = [base_seg]
else:
    for seg in st.session_state.segments:
        ensure_time_defaults(seg)
if "stream_state" not in st.session_state:
    st.session_state.stream_state = "idle"  # idle, running, paused
if "stream_time_index" not in st.session_state:
    st.session_state.stream_time_index = 0  # integer index in combined arrays
if "final_output_mode" not in st.session_state:
    st.session_state.final_output_mode = None
if "final_output_ready" not in st.session_state:
    st.session_state.final_output_ready = False

# Seed + mode
with sidebar_card("Simulation Setup"):
    seed_value = st.sidebar.number_input("Random Seed (optional)", value=-1, step=1)
    if seed_value >= 0:
        np.random.seed(int(seed_value)); random.seed(int(seed_value))
    mode = st.sidebar.selectbox("Simulation Mode", ["1D Expression", "Straight Line", "2D Expression"])

# Noise and plot params
with sidebar_card("Noise & Plot"):
    noise_type = st.sidebar.selectbox("Noise Type", ["None", "Gaussian", "Uniform"])
    noise_strength = st.sidebar.slider("Noise Strength", 0.0, 5.0, 0.4)
    color_cols = st.sidebar.columns(2)
    with color_cols[0]:
        color_cols[0].markdown("<div class='expr-label small'>Clean Color</div>", unsafe_allow_html=True)
        clean_color = color_cols[0].color_picker(
            "Clean Color",
            "#1f77b4",
            key="clean_color_picker",
            label_visibility="collapsed",
        )
    with color_cols[1]:
        color_cols[1].markdown("<div class='expr-label small'>Noisy Color</div>", unsafe_allow_html=True)
        noisy_color = color_cols[1].color_picker(
            "Noisy Color",
            "#ff7f0e",
            key="noisy_color_picker",
            label_visibility="collapsed",
        )
    clean_linewidth = st.sidebar.slider("Clean Thickness", 0.5, 5.0, 1.0, 0.1)
    noisy_linewidth = st.sidebar.slider("Noisy Thickness", 0.5, 5.0, 1.5, 0.1)
    show_data = st.sidebar.checkbox("Show Data Table", False)

# DB toggle
with sidebar_card("Ingestion Mode"):
    db_mode = st.sidebar.radio("Ingestion Mode", ["Mock Only", "PostgreSQL"])
    db_available = False
    if db_mode == "PostgreSQL":
        ok, missing = db_config_valid(my_secrets)
        if not ok:
            st.sidebar.error(f"DB config missing key: {missing}. Check your secrets.")
            st.sidebar.info("Falling back to Mock Only mode until secrets are fixed.")
            db_available = False
        else:
            try:
                create_table_if_not_exists()
                st.sidebar.success("Connected to PostgreSQL (table ready)."); db_available = True
            except Exception as e:
                st.sidebar.error(f"Failed DB init: {e}"); db_available = False
    else:
        st.sidebar.info("Mock Only mode")

# -------------------------
# Dynamic expression UI (aligned inputs with remove button)
# -------------------------
st.markdown("### Multi-expression configuration")

if mode == "1D Expression":
    st.markdown("Add 1D expressions — they will be concatenated sequentially.")
    for i, item in enumerate(st.session_state.expr_list):
        label = f"{ordinal(i+1)} exp"
        cols = st.columns([0.17, 0.71, 0.12])
        cols[0].markdown(f"<div class='expr-label'>{label}</div>", unsafe_allow_html=True)
        new_val = cols[1].text_input(
            f"expr_{i}",
            value=item["expr"],
            key=f"expr1_{i}",
            label_visibility="collapsed",
            placeholder="Enter expression f(t)"
        )
        st.session_state.expr_list[i]["expr"] = new_val
        render_time_inputs_ui(st.session_state.expr_list[i], f"expr_{i}")
        if cols[2].button("🗑", key=f"remove1_{i}", help="Remove this expression", type="secondary", use_container_width=True):
            if len(st.session_state.expr_list) > 1:
                st.session_state.expr_list.pop(i)
                st.experimental_rerun()
    if st.button("Add Expression", key="add_expr_btn", type="primary"):
        new_item = {"expr": "t * sin(0.3*t) + 3"}
        ensure_time_defaults(new_item)
        st.session_state.expr_list.append(new_item)
        st.experimental_rerun()

elif mode == "Straight Line":
    st.markdown("Add straight-line segments (A → B). Consecutive segments will connect end-to-end.")
    for i, seg in enumerate(st.session_state.segments):
        label = f"{ordinal(i+1)} segment"
        # label, ax, ay, bx, by, remove
        cols = st.columns([0.12, 0.18, 0.18, 0.18, 0.24])
        cols[0].markdown(f"**{label}**")
        ax = cols[1].number_input(f"A.x_{i}", value=float(seg["ax"]), key=f"ax_{i}")
        ay = cols[2].number_input(f"A.y_{i}", value=float(seg["ay"]), key=f"ay_{i}")
        bx = cols[3].number_input(f"B.x_{i}", value=float(seg["bx"]), key=f"bx_{i}")
        by = cols[4].number_input(f"B.y_{i}", value=float(seg["by"]), key=f"by_{i}")
        st.session_state.segments[i].update({"ax": ax, "ay": ay, "bx": bx, "by": by})
        render_time_inputs_ui(st.session_state.segments[i], f"seg_{i}")
        # remove button on new row aligned to right
        rem_cols = st.columns([0.88, 0.12])
        if rem_cols[1].button("🗑", key=f"remseg_{i}", type="primary"):
            if len(st.session_state.segments) > 1:
                st.session_state.segments.pop(i)
                st.experimental_rerun()
    if st.button("Add Segment", key="add_seg_btn", type="primary"):
        new_seg = {"ax": 0.0, "ay": 0.0, "bx": 10.0, "by": 5.0}
        ensure_time_defaults(new_seg)
        st.session_state.segments.append(new_seg)
        st.experimental_rerun()

elif mode == "2D Expression":
    st.markdown("Add parametric (x(t), y(t)) expression pairs.")
    for i, item in enumerate(st.session_state.expr2d_list):
        label = f"{ordinal(i+1)} exp"
        cols = st.columns([0.16, 0.32, 0.32, 0.20])
        cols[0].markdown(f"<div class='expr-label'>{label}</div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<div class='expr-label small'>x(t)</div>", unsafe_allow_html=True)
            new_x = st.text_input(
                f"x_expr_{i}",
                value=item["x"],
                key=f"xexpr_{i}",
                label_visibility="collapsed",
                placeholder="Enter x(t)"
            )
        with cols[2]:
            st.markdown("<div class='expr-label small'>y(t)</div>", unsafe_allow_html=True)
            new_y = st.text_input(
                f"y_expr_{i}",
                value=item["y"],
                key=f"yexpr_{i}",
                label_visibility="collapsed",
                placeholder="Enter y(t)"
            )
        st.session_state.expr2d_list[i]["x"] = new_x
        st.session_state.expr2d_list[i]["y"] = new_y
        render_time_inputs_ui(st.session_state.expr2d_list[i], f"expr2d_{i}")
        if cols[3].button("🗑", key=f"rem2d_{i}", help="Remove this expression pair", type="secondary", use_container_width=True):
            if len(st.session_state.expr2d_list) > 1:
                st.session_state.expr2d_list.pop(i)
                st.experimental_rerun()
    if st.button("Add 2D Expression", key="add_2d_btn", type="primary"):
        new_expr = {"x": "10*sin(0.4*t)", "y": "10*cos(0.4*t)"}
        ensure_time_defaults(new_expr)
        st.session_state.expr2d_list.append(new_expr)
        st.experimental_rerun()

# -------------------------
# Build combined series for selected mode
# -------------------------
def build_combined_1d():
    clean_segments = []
    noisy_segments = []
    time_segments = []
    offset = 0.0
    for i, item in enumerate(st.session_state.expr_list):
        label = f"{ordinal(i+1)} expression"
        t_eval, t_global, offset = generate_time_series(item, offset, label)
        y_clean = eval_expr(t_eval, item["expr"])
        if noise_type == "Gaussian":
            y_noisy = y_clean + gaussian_noise(len(t_eval), noise_strength)
        elif noise_type == "Uniform":
            y_noisy = y_clean + uniform_noise(len(t_eval), noise_strength)
        else:
            y_noisy = y_clean
        clean_segments.append(y_clean)
        noisy_segments.append(y_noisy)
        time_segments.append(t_global)
    times = np.concatenate(time_segments) if time_segments else np.array([])
    clean = np.concatenate(clean_segments) if clean_segments else np.array([])
    noisy = np.concatenate(noisy_segments) if noisy_segments else np.array([])
    return times, clean, noisy

def build_combined_straight():
    clean_x_parts = []
    clean_y_parts = []
    noisy_y_parts = []
    time_parts = []
    offset = 0.0
    for i, seg in enumerate(st.session_state.segments):
        label = f"{ordinal(i+1)} segment"
        t_eval, t_global, offset = generate_time_series(seg, offset, label)
        pts = len(t_eval)
        x_vals = np.linspace(seg["ax"], seg["bx"], pts)
        y_vals = np.linspace(seg["ay"], seg["by"], pts)
        if noise_type == "Gaussian":
            y_noisy = y_vals + gaussian_noise(pts, noise_strength)
        elif noise_type == "Uniform":
            y_noisy = y_vals + uniform_noise(pts, noise_strength)
        else:
            y_noisy = y_vals
        clean_x_parts.append(x_vals)
        clean_y_parts.append(y_vals)
        noisy_y_parts.append(y_noisy)
        time_parts.append(t_global)
    times = np.concatenate(time_parts) if time_parts else np.array([])
    clean_x = np.concatenate(clean_x_parts) if clean_x_parts else np.array([])
    clean_y = np.concatenate(clean_y_parts) if clean_y_parts else np.array([])
    noisy_y = np.concatenate(noisy_y_parts) if noisy_y_parts else np.array([])
    return times, clean_x, clean_y, noisy_y

def build_combined_2d():
    x_parts = []; y_parts = []; noisy_x_parts = []; noisy_y_parts = []; time_parts = []
    offset = 0.0
    for i, item in enumerate(st.session_state.expr2d_list):
        label = f"{ordinal(i+1)} 2D expression"
        t_eval, t_global, offset = generate_time_series(item, offset, label)
        x_clean = eval_expr(t_eval, item["x"])
        y_clean = eval_expr(t_eval, item["y"])
        if noise_type == "Gaussian":
            x_noisy = x_clean + gaussian_noise(len(t_eval), noise_strength)
            y_noisy = y_clean + gaussian_noise(len(t_eval), noise_strength)
        elif noise_type == "Uniform":
            x_noisy = x_clean + uniform_noise(len(t_eval), noise_strength)
            y_noisy = y_clean + uniform_noise(len(t_eval), noise_strength)
        else:
            x_noisy = x_clean; y_noisy = y_clean
        x_parts.append(x_clean); y_parts.append(y_clean)
        noisy_x_parts.append(x_noisy); noisy_y_parts.append(y_noisy)
        time_parts.append(t_global)
    times = np.concatenate(time_parts) if time_parts else np.array([])
    clean_x = np.concatenate(x_parts) if x_parts else np.array([])
    clean_y = np.concatenate(y_parts) if y_parts else np.array([])
    noisy_x = np.concatenate(noisy_x_parts) if noisy_x_parts else np.array([])
    noisy_y = np.concatenate(noisy_y_parts) if noisy_y_parts else np.array([])
    return times, clean_x, clean_y, noisy_x, noisy_y

# Build arrays and render plots; capture csv/png bytes to show download buttons side-by-side
if mode == "1D Expression":
    combined_time, combined_clean, combined_noisy = build_combined_1d()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(combined_time, combined_clean, label="Clean", linewidth=clean_linewidth, color=clean_color)
    ax.legend(); ax.grid(True); ax.set_title("Multi 1D Expression (Clean Preview)")
    ax.set_xlabel("time (s)")
    st.pyplot(fig)

    if show_data:
        preview_df = pd.DataFrame({"time_seconds": combined_time, "clean_y": combined_clean})
        st.dataframe(preview_df)
    stream_primary = combined_clean
    stream_noisy_primary = combined_noisy
    stream_secondary=None
    stream_noisy_secondary=None

elif mode == "Straight Line":
    combined_time, clean_x, clean_y, noisy_y = build_combined_straight()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(combined_time, clean_y, label="Clean", linewidth=clean_linewidth, color=clean_color)
    ax.legend(); ax.grid(True); ax.set_title("Multi Straight Line (Clean Preview)")
    st.pyplot(fig)
    if show_data:
        df_line = pd.DataFrame({"time_seconds": combined_time, "clean_x": clean_x, "clean_y": clean_y})
        st.dataframe(df_line)
    stream_primary = clean_y
    stream_noisy_primary = noisy_y
    stream_secondary = clean_x
    stream_noisy_secondary = None

elif mode == "2D Expression":
    combined_time, clean_x, clean_y, noisy_x, noisy_y = build_combined_2d()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(clean_x, clean_y, label="Clean", linewidth=clean_linewidth, color=clean_color)
    ax.legend(); ax.grid(True); ax.set_title("Multi 2D Expression (Clean Preview)")
    st.pyplot(fig)
    if show_data:
        df2 = pd.DataFrame({"time_seconds": combined_time, "clean_x": clean_x, "clean_y": clean_y})
        st.dataframe(df2)
    stream_primary = clean_y
    stream_noisy_primary = noisy_y
    stream_secondary = clean_x
    stream_noisy_secondary = noisy_x

# init streaming state variables
if "stream_array_len" not in st.session_state:
    st.session_state.stream_array_len = len(stream_primary) if stream_primary is not None else 0
# If user changes exprs/time, reset index
if st.session_state.stream_time_index >= (len(stream_primary) if stream_primary is not None else 0):
    st.session_state.stream_time_index = 0

# -------------------------
# Streaming control UI (styled & side-by-side)
# -------------------------
st.markdown('<div class="stream-control">', unsafe_allow_html=True)
cols = st.columns([1,1])
left = cols[0]
right = cols[1]

# Show attractive buttons using normal Streamlit buttons but styled via CSS classes for visuals:
# We can't directly assign CSS class to st.button; instead, we use columns + markdown to emulate width and use buttons.
with left:
    if st.session_state.stream_state == "idle":
        if st.button("Start Streaming", key="start_stream_btn", type="primary"):
            st.session_state.stream_state = "running"
            st.session_state.stream_time_index = 0
            st.session_state.final_output_mode = None
            st.session_state.final_output_ready = False
            for k in ["streamed_times","streamed_clean_primary","streamed_noisy_primary","streamed_clean_secondary","streamed_noisy_secondary"]:
                if k in st.session_state:
                    del st.session_state[k]
    elif st.session_state.stream_state == "running":
        if st.button("Pause Streaming", key="pause_stream_btn", type="primary"):
            st.session_state.stream_state = "paused"
    elif st.session_state.stream_state == "paused":
        if st.button("Resume Streaming", key="resume_stream_btn", type="primary"):
            st.session_state.stream_state = "running"

with right:
    if st.button("Stop Streaming", key="stop_stream_button", type="primary"):
        st.session_state.stream_state = "idle"
        st.session_state.stream_time_index = 0
        st.session_state.final_output_mode = None
        st.session_state.final_output_ready = False
        for k in ["streamed_times","streamed_clean_primary","streamed_noisy_primary","streamed_clean_secondary","streamed_noisy_secondary"]:
            if k in st.session_state:
                del st.session_state[k]
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Streaming loop (responds to stream_state)
# -------------------------
if st.session_state.stream_state == "running":
    st.subheader("Real-Time Streaming")
    time_arr = np.asarray(combined_time, dtype=float)
    clean_primary_arr = np.asarray(stream_primary, dtype=float)
    noisy_primary_arr = np.asarray(stream_noisy_primary, dtype=float) if stream_noisy_primary is not None else None
    clean_secondary_arr = np.asarray(stream_secondary, dtype=float) if stream_secondary is not None else None
    noisy_secondary_arr = np.asarray(stream_noisy_secondary, dtype=float) if stream_noisy_secondary is not None else None

    streamed_times = st.session_state.get("streamed_times", [])
    streamed_clean_primary = st.session_state.get("streamed_clean_primary", [])
    streamed_noisy_primary = st.session_state.get("streamed_noisy_primary", [])
    streamed_clean_secondary = st.session_state.get("streamed_clean_secondary", [])
    streamed_noisy_secondary = st.session_state.get("streamed_noisy_secondary", [])

    plot_placeholder = st.empty()
    progress = st.progress(0)

    # adjust start_clock so wall-clock sync matches resumed index
    idx = st.session_state.stream_time_index
    total_len = len(time_arr)
    current_time_value = time_arr[idx] if total_len and idx < total_len else 0.0
    start_clock = pytime.time() - current_time_value
    sent = idx

    while idx < total_len and st.session_state.stream_state == "running":
        t_val = float(time_arr[idx])
        c_primary = float(clean_primary_arr[idx])
        noisy_primary_val = float(noisy_primary_arr[idx]) if noisy_primary_arr is not None else c_primary
        c_secondary = float(clean_secondary_arr[idx]) if clean_secondary_arr is not None else None
        noisy_secondary_val = float(noisy_secondary_arr[idx]) if noisy_secondary_arr is not None else c_secondary

        streamed_times.append(t_val)
        streamed_clean_primary.append(c_primary)
        streamed_noisy_primary.append(noisy_primary_val)
        if c_secondary is not None:
            streamed_clean_secondary.append(c_secondary)
            streamed_noisy_secondary.append(noisy_secondary_val)

        record = {
            "time_value": t_val,
            "clean_y": c_primary,
            "noisy_y": noisy_primary_val,
            "noise_value": (noisy_primary_val - c_primary),
            "clean_x": c_secondary,
            "noisy_x": noisy_secondary_val,
            "noise_value_x": (noisy_secondary_val - c_secondary) if c_secondary is not None else None,
            "noise_value_y": (noisy_primary_val - c_primary),
        }

        if db_mode == "Mock Only" or not db_available:
            mock_send_to_db(record)
        else:
            try:
                insert_record_pg(record)
            except Exception as e:
                st.error(f"DB Insert Failed: {e}")
                st.session_state.stream_state = "idle"
                st.session_state.stream_time_index = 0
                st.session_state.final_output_mode = None
                st.session_state.final_output_ready = False
                break

        sent += 1
        progress.progress(int((sent / max(1, total_len)) * 100))

        # live plot update (use the global color & linewidth settings)
        if mode == "2D Expression":
            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(streamed_clean_secondary, streamed_clean_primary, label="Clean", color=clean_color, linewidth=clean_linewidth)
            if len(streamed_noisy_secondary):
                ax.plot(streamed_noisy_secondary, streamed_noisy_primary, label="Noisy", color=noisy_color, linewidth=noisy_linewidth)
            ax.legend(); ax.grid(True)
        else:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(streamed_times, streamed_clean_primary, label="Clean", color=clean_color, linewidth=clean_linewidth)
            ax.plot(streamed_times, streamed_noisy_primary, label="Noisy", color=noisy_color, linewidth=noisy_linewidth)
            ax.set_xlabel("time"); ax.set_ylabel("y(t)")
            ax.legend(); ax.grid(True)

        plot_placeholder.pyplot(fig)
        plt.close(fig)

        # advance index and record in session state
        st.session_state.stream_time_index = idx + 1
        idx += 1

        # wall-clock sync with small chunks so Pause is responsive
        if idx < total_len:
            elapsed = pytime.time() - start_clock
            target_elapsed = time_arr[idx]
            sleep_time = target_elapsed - elapsed
            if sleep_time > 0:
                waited = 0.0
                while waited < sleep_time:
                    if st.session_state.stream_state != "running":
                        break
                    chunk = min(0.1, sleep_time - waited)
                    pytime.sleep(chunk)
                    waited += chunk
                if st.session_state.stream_state != "running":
                    break

    # save streamed arrays back to session_state for pause/resume
    st.session_state["streamed_times"] = streamed_times
    st.session_state["streamed_clean_primary"] = streamed_clean_primary
    st.session_state["streamed_noisy_primary"] = streamed_noisy_primary
    st.session_state["streamed_clean_secondary"] = streamed_clean_secondary
    st.session_state["streamed_noisy_secondary"] = streamed_noisy_secondary

    if total_len > 0 and st.session_state.stream_time_index >= total_len:
        st.session_state.stream_state = "idle"
        st.session_state.stream_time_index = 0
        st.session_state.final_output_mode = mode
        st.session_state.final_output_ready = True
        st.success("Streaming finished.")

elif st.session_state.stream_state == "paused":
    st.info("Streaming paused. Click Resume to continue.")
else:
    st.info("Stream is idle. Configure simulation and press Start Streaming.")

final_mode = st.session_state.get("final_output_mode")
final_ready = st.session_state.get("final_output_ready", False)
stored_times = st.session_state.get("streamed_times", [])
if final_ready and final_mode and stored_times:
    st.subheader("Final Output (with Noise)")
    final_times = np.asarray(stored_times, dtype=float)
    clean_primary_final = np.asarray(st.session_state.get("streamed_clean_primary", []), dtype=float)
    noisy_primary_final = np.asarray(st.session_state.get("streamed_noisy_primary", []), dtype=float)
    clean_secondary_list = st.session_state.get("streamed_clean_secondary", [])
    noisy_secondary_list = st.session_state.get("streamed_noisy_secondary", [])
    clean_secondary_final = np.asarray(clean_secondary_list, dtype=float) if len(clean_secondary_list) else np.array([])
    noisy_secondary_final = np.asarray(noisy_secondary_list, dtype=float) if len(noisy_secondary_list) else np.array([])

    if final_mode == "2D Expression":
        fig, ax = plt.subplots(figsize=(6,6))
        x_clean = clean_secondary_final if clean_secondary_final.size else np.zeros_like(final_times)
        x_noisy = noisy_secondary_final if noisy_secondary_final.size else x_clean
        ax.plot(x_clean, clean_primary_final, label="Clean", color=clean_color, linewidth=clean_linewidth)
        ax.plot(x_noisy, noisy_primary_final, label="Noisy", color=noisy_color, linewidth=noisy_linewidth)
        ax.legend(); ax.grid(True)
        df_final = pd.DataFrame({
            "time_seconds": final_times,
            "clean_x": x_clean,
            "clean_y": clean_primary_final,
            "noisy_x": x_noisy,
            "noisy_y": noisy_primary_final,
        })
    elif final_mode == "Straight Line":
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(final_times, clean_primary_final, label="Clean", color=clean_color, linewidth=clean_linewidth)
        ax.plot(final_times, noisy_primary_final, label="Noisy", color=noisy_color, linewidth=noisy_linewidth)
        ax.legend(); ax.grid(True); ax.set_xlabel("time (s)"); ax.set_ylabel("y(t)")
        df_final = pd.DataFrame({
            "time_seconds": final_times,
            "clean_x": clean_secondary_final if clean_secondary_final.size else np.zeros_like(final_times),
            "clean_y": clean_primary_final,
            "noisy_y": noisy_primary_final,
        })
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(final_times, clean_primary_final, label="Clean", color=clean_color, linewidth=clean_linewidth)
        ax.plot(final_times, noisy_primary_final, label="Noisy", color=noisy_color, linewidth=noisy_linewidth)
        ax.legend(); ax.grid(True); ax.set_xlabel("time (s)"); ax.set_ylabel("y(t)")
        df_final = pd.DataFrame({
            "time_seconds": final_times,
            "clean_y": clean_primary_final,
            "noisy_y": noisy_primary_final,
        })

    st.pyplot(fig)
    png_buf = BytesIO(); fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight"); png_buf.seek(0); png_bytes = png_buf.getvalue()
    plt.close(fig)
    csv_buf = BytesIO(); df_final.to_csv(csv_buf, index=False); csv_bytes = csv_buf.getvalue()

    slug = final_mode.lower().replace(" ", "_")
    c1, c2 = st.columns(2)
    with c1:
        render_download_button("Download CSV", csv_bytes, f"final_{slug}.csv", "text/csv", f"final_csv_{slug}")
    with c2:
        render_download_button("Download Plot PNG", png_bytes, f"final_{slug}.png", "image/png", f"final_png_{slug}")