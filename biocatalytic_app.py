import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Biocatalytic Simulation App", layout="wide")
st.title("🧪 Biocatalytic Process Simulator")

# Sidebar parameters
st.sidebar.header("🔧 Simulation Parameters")
mode = st.sidebar.selectbox("Select Mode", ["batch", "fed-batch", "continuous"])
TK_conc = st.sidebar.number_input("TK Concentration (mM)", value=0.00032)
TA_conc = st.sidebar.number_input("TA Concentration (mM)", value=0.00032)
HPA0 = st.sidebar.number_input("Initial HPA (mM)", value=100)
GA0 = st.sidebar.number_input("Initial GA (mM)", value=100)
MBA0 = st.sidebar.number_input("Initial MBA (mM)", value=50)
feed_HPA = st.sidebar.number_input("Feed HPA (mM)", value=200)
feed_GA = st.sidebar.number_input("Feed GA (mM)", value=200)
flow_rate = st.sidebar.number_input("Flow Rate (mL/min)", value=0.1)
time_span = st.sidebar.slider("Total Time (min)", 10, 300, 120)

reactor_volume = 1.0
ERY0 = 0
AP0 = 0
ABT0 = 0
AR0 = 1

Ka_TK = 13.2
Kia_TK = 42.2
Kb_TK = 16.1
Kib_TK = 597.6
Kiq_TK = 565.8
Kcat_TK = 5076
Km_MBA = 0.51
Km_ERY = 95.5
Km_ABT = 37.0
Km_AP = 16.1
kf = 95.1
kr = 12.0
Keq = 843

def model(t, y):
    HPA, GA, ERY, MBA, AP, ABT, AR = y
    Vmax_TK = Kcat_TK * TK_conc * AR
    num_TK = Vmax_TK * HPA * GA
    denom_TK = (Kb_TK * HPA * (1 + HPA / Kia_TK) +
                Ka_TK * GA * (1 + GA / Kib_TK) +
                HPA * GA + (Ka_TK / Kiq_TK) * GA * ERY +
                (Ka_TK * Kib_TK / Kiq_TK) * ERY)
    v_TK = num_TK / denom_TK
    Ktox = 483.3 * GA / (7.357e7 + GA)
    dAR_dt = -Ktox * AR
    Vmax_TA = kf * kr * TA_conc
    driving_force = (MBA * ERY - (ABT * AP) / Keq)
    denom_TA = (Km_MBA * Km_ERY + Km_ERY * MBA + Km_MBA * ERY + MBA * ERY)
    v_TA = Vmax_TA * driving_force / denom_TA

    dHPA_dt = -v_TK
    dGA_dt = -v_TK
    dERY_dt = v_TK - v_TA
    dMBA_dt = -v_TA
    dAP_dt = v_TA
    dABT_dt = v_TA

    if mode == 'fed-batch':
        dHPA_dt += (flow_rate / reactor_volume) * (feed_HPA - HPA)
        dGA_dt += (flow_rate / reactor_volume) * (feed_GA - GA)
    elif mode == 'continuous':
        dHPA_dt += (flow_rate / reactor_volume) * (feed_HPA - HPA)
        dGA_dt += (flow_rate / reactor_volume) * (feed_GA - GA)
        dERY_dt -= (flow_rate / reactor_volume) * ERY
        dMBA_dt -= (flow_rate / reactor_volume) * MBA
        dAP_dt -= (flow_rate / reactor_volume) * AP
        dABT_dt -= (flow_rate / reactor_volume) * ABT

    return [dHPA_dt, dGA_dt, dERY_dt, dMBA_dt, dAP_dt, dABT_dt, dAR_dt]

if st.button("▶ Run Simulation"):
    y0 = [HPA0, GA0, ERY0, MBA0, AP0, ABT0, AR0]
    t_eval = np.linspace(0, time_span, 300)
    sol = solve_ivp(model, (0, time_span), y0, t_eval=t_eval)

    df = pd.DataFrame({
        'Time (min)': sol.t,
        '[HPA]': sol.y[0],
        '[GA]': sol.y[1],
        '[ERY]': sol.y[2],
        '[MBA]': sol.y[3],
        '[AP]': sol.y[4],
        '[ABT]': sol.y[5],
        'Enzyme Activity': sol.y[6]
    })

    yield_final = sol.y[5][-1] / MBA0 * 100
    st.write(f"**Final ABT Yield:** {yield_final:.2f}%")
    st.write(f"**Final ABT Concentration:** {sol.y[5][-1]:.2f} mM")

    # Plot ERY
    fig1, ax1 = plt.subplots()
    ax1.plot(sol.t, sol.y[2], label='[ERY]', color='blue')
    ax1.set_title("Production of ERY")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Concentration (mM)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot ABT
    fig2, ax2 = plt.subplots()
    ax2.plot(sol.t, sol.y[5], label='[ABT]', color='green')
    ax2.set_title("Formation of ABT")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Concentration (mM)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv, file_name="results.csv", mime="text/csv")
