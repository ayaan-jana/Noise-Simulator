# Noise Simulator — Multi-Expression Streaming Tool

A Streamlit-based real-time simulation platform supporting **multi-expression function plotting**, **parametric 2D motion**, **straight-line path segments**, **noise injection**, **real-time streaming**, **pause/resume**, and **optional PostgreSQL ingestion**.

---

## Features

### Simulation Modes

| Mode              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **1D Expression** | Sequential expressions combined into a single curve `y(t)` vs time.        |
| **Straight Line** | Multiple linear segments between points `(Ax, Ay) → (Bx, By)`.             |
| **2D Expression** | Parametric curves using expressions for `x(t)` and `y(t)`.                 |

---

### Noise Models

| Type     | Description                                     |
|----------|-------------------------------------------------|
| None     | Clean waveform, no noise added.                 |
| Gaussian | Normal random noise added to values.            |
| Uniform  | Random noise in a symmetric amplitude interval. |

---

### Real-Time Streaming

- Start / Pause / Resume / Stop controls.
- Live updating graphs.
- Adjustable time unit (**Milliseconds** or **Seconds**).
- Configurable:
  - Start time
  - End time
  - Time interval
- Optional PostgreSQL ingestion:
  - Each streamed point can be inserted into a `noise_stream` table.
  - Fallback **Mock mode** that prints records to the console if DB is not configured.
- Progress bar indicating streaming completion percentage.

---

### Data Export

For each mode, the app provides:

- **Download CSV** of generated data.
- **Download PNG** of the plotted curve/path.

Data columns vary by mode:

- **1D Expression**
  - `time`, `clean_y`, `noisy_y`
- **Straight Line**
  - `time`, `clean_x`, `clean_y`, `noisy_y`
- **2D Expression**
  - `time`, `clean_x`, `clean_y`, `noisy_x`, `noisy_y`

---

### UI & UX

- Multi-expression configuration for:
  - 1D expressions
  - 2D parametric expressions
  - Straight-line segments
- Dynamic add/remove:
  - Add new expressions or segments using “Add Expression” / “Add Segment” buttons.
  - Remove entries using a **trash icon** button.
- Custom CSS for:
  - Styled primary action buttons (Start, Pause/Resume, Stop).
  - Improved alignment and spacing of controls.
- Optional data table view for inspecting underlying data.

---

## Installation & Setup

### Requirements

- Python 3.9+
- (Optional) PostgreSQL instance if you want to enable DB ingestion.

### Install Dependencies

```bash
pip install streamlit numpy pandas matplotlib psycopg2-binary toml
