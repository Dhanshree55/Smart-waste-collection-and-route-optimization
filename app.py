import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# FIXED CSS (TITLE VISIBLE)
# -----------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0b1c24;
    color: white !important;
}

.title {
    font-size: 34px;
    font-weight: bold;
    color: white !important;
    margin-bottom: 10px;
}

.card {
    background-color: #112a33;
    padding: 15px;
    border-radius: 10px;
    color: white;
}

.metric {
    font-size: 22px;
    font-weight: bold;
    color: #f4c430;
}

.small {
    color: #9fb3c8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER (NOW ALWAYS VISIBLE)
# -----------------------------
st.markdown('<div class="title">♻️ Smart Waste Management Dashboard</div>', unsafe_allow_html=True)
st.caption("AI-driven Monitoring • Route Optimization • Forecasting")

# -----------------------------
# DATA (SIMULATION)
# -----------------------------
np.random.seed(42)

n_points = 300
lat = np.random.uniform(19.9, 20.1, n_points)
lon = np.random.uniform(73.7, 74.0, n_points)

df = pd.DataFrame({
    "lat": lat,
    "lon": lon
})

# select subset as "optimized route"
selected_idx = np.random.choice(n_points, 50, replace=False)
route = df.iloc[selected_idx]

# -----------------------------
# ML MODEL (FOR KPI ONLY)
# -----------------------------
df_ml = pd.DataFrame({
    "time": np.arange(n_points),
    "waste": np.random.randint(400, 1200, n_points)
})

model = RandomForestRegressor()
model.fit(df_ml[["time"]], df_ml["waste"])

future_pred = model.predict([[n_points + 10]])[0]

# -----------------------------
# KPI ROW
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f'<div class="card"><div>Total Waste</div><div class="metric">{df_ml.waste.sum()}</div></div>', unsafe_allow_html=True)

c2.markdown(f'<div class="card"><div>Avg Waste</div><div class="metric">{round(df_ml.waste.mean(),1)}</div></div>', unsafe_allow_html=True)

c3.markdown(f'<div class="card"><div>Route Efficiency</div><div class="metric">43%</div></div>', unsafe_allow_html=True)

c4.markdown(f'<div class="card"><div>Forecast Waste</div><div class="metric">{round(future_pred,1)}</div></div>', unsafe_allow_html=True)

# -----------------------------
# MAIN LAYOUT
# -----------------------------
left, right = st.columns([3,1])

# -----------------------------
# ROUTE OPTIMIZATION GRAPH (REPLACED)
# -----------------------------
with left:
    st.markdown("### 🚛 Optimized Waste Collection Route")

    fig, ax = plt.subplots(figsize=(10,5))

    # all bins
    ax.scatter(df["lon"], df["lat"], alpha=0.3, label="All Bins")

    # selected bins
    ax.scatter(route["lon"], route["lat"], color="red", label="Selected Bins")

    # route line
    ax.plot(route["lon"], route["lat"], color="blue", linewidth=2, label="Optimized Route")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    st.pyplot(fig)

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:
    st.markdown('<div class="card"><div>Carbon Footprint</div><div class="metric">1436</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div>Segregation Rate</div><div class="metric">43%</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div>Electricity Generated</div><div class="metric">779 kWh</div></div>', unsafe_allow_html=True)

# -----------------------------
# BOTTOM SECTION
# -----------------------------
b1, b2 = st.columns([2,1])

with b1:
    st.markdown("### 🛠 Issues Tracker")

    issues = pd.DataFrame({
        "Issue": ["Sensor Fault", "Bin Overflow", "Route Delay", "Collection Missed"],
        "Status": ["Open", "Resolved", "In Progress", "Open"],
        "Type": ["System", "System", "Logistics", "System"]
    })

    st.dataframe(issues)

with b2:
    st.markdown("### 📊 Profit Analysis")

    fig2, ax2 = plt.subplots()
    ax2.bar(range(1,13), np.random.randint(500,1500,12))
    st.pyplot(fig2)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI Smart Waste System • Portfolio Project")