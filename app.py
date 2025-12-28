# app.py - FIXED v4.0 (Copy entire file to GitHub!)
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="LEHS Dashboard v4.0")

# ==============================================================================
# LEHS COEFFICIENTS - DEFINED FIRST!
# ==============================================================================
practices = {
    "Rice_AWD": {"CH4": 0.24, "Income_ha": 8000, "Jobs_ha": 0.0235, "DALY_tCO2e": 6.8, "source": "IRRI Punjab"},
    "Rice_DSR": {"CH4": 0.30, "Income_ha": 13789, "Jobs_ha": 0.0235, "DALY_tCO2e": 6.8, "source": "IRRI Punjab"},
    "Dairy_Feed": {"CH4": 0.18, "Income_cow": 102188, "Jobs_cow": 0.05, "DALY_tCO2e": 6.8, "source": "NDDB 2M cows"},
    "Dairy_AS": {"CH4": 0.06, "Income_cow": 1300, "Jobs_cow": 0.03, "DALY_tCO2e": 3.4, "source": "NDDB"}
}

@st.cache_data
def calculate_lehs_production(sector, practice, states, small_farms, med_farms, large_farms, ch4_intensity, adoption_rate):
    """Production-grade LEHS calculator"""
    
    # ERROR HANDLING
    total_farms = small_farms + med_farms + large_farms
    if total_farms == 0:
        return pd.DataFrame(), "❌ Select at least 1 farm!", {}
    if not states:
        return pd.DataFrame(), "❌ Select at least 1 state!", {}
    
    # FARM SIZES (BAHS 2023)
    farm_sizes = {"Rice": [1.2, 3.5, 10.0], "Dairy": [3.2, 9.8, 28.4]}
    sizes = farm_sizes[sector]
    total_scale = sizes[0]*small_farms + sizes[1]*med_farms + sizes[2]*large_farms
    
    # STATE WEIGHTS (PLFS 2024)
    state_weights = {"Punjab": 0.08, "Haryana": 0.12, "UP": 0.35, "Gujarat": 0.10, "Bihar": 0.15, "WB": 0.12, "Telangana": 0.08}
    
    coeff = practices[practice]
    
    # MONTE CARLO SIMULATION
    n_sim = 100
    ch4_sim, income_sim, dalys_sim, jobs_sim = [], [], [], []
    
    for _ in range(n_sim):
        efficacy = coeff["CH4"] * np.random.normal(1, 0.15)
        scale_var = total_scale * np.random.normal(1, 0.10)
        
        baseline = ch4_intensity * scale_var * (9*365 if sector=="Dairy" else 1)
        ch4 = baseline * efficacy * adoption_rate * 0.028
        
        dalys = ch4 * coeff["DALY_tCO2e"]
        
        if sector == "Rice":
            income = coeff["Income_ha"] * scale_var * adoption_rate / 1e7
            jobs = scale_var * coeff["Jobs_ha"]
        else:
            income = coeff["Income_cow"] * scale_var * adoption_rate / 1e7
            jobs = scale_var * coeff["Jobs_cow"]
        
        ch4_sim.append(ch4)
        income_sim.append(income)
        dalys_sim.append(dalys)
        jobs_sim.append(jobs)
    
    # P10/P50/P90
    results = {
        "CH4_P10": np.percentile(ch4_sim, 10),
        "CH4_P50": np.percentile(ch4_sim, 50),
        "CH4_P90": np.percentile(ch4_sim, 90),
        "Income_P50": np.percentile(income_sim, 50),
        "DALYs_P50": np.percentile(dalys_sim, 50),
        "Jobs_P50": np.percentile(jobs_sim, 50)
    }
    
    return results, "✅ SUCCESS", state_weights

# ==============================================================================
# DASHBOARD UI
# ==============================================================================
st.markdown("# 🌾 LEHS Co-Benefits Calculator **v4.0** ⭐")
st.markdown("*Live Monte Carlo | BAHS 2023 | NDDB | IRRI | GBD-MAPS*")

# INPUTS
col1, col2 = st.columns([3,1])
with col1:
    sector = st.selectbox("Sector", ["Rice", "Dairy"])
    practice = st.selectbox("Practice", list(practices.keys()))
    
with col2:
    adoption_rate = st.slider("Adoption %", 10, 50, 25)

st.subheader("🌍 States")
states = st.multiselect("Select states", 
    ["Punjab", "Haryana", "Gujarat", "UP", "Bihar", "WB", "Telangana"], 
    default=["UP", "Haryana"])

st.subheader("👨‍🌾 Farms (BAHS 2023)")
col1, col2, col3 = st.columns(3)
small_farms = col1.number_input("Smallholder", 1000, 0, 10000)
med_farms = col2.number_input("Medium", 500, 0, 5000)
large_farms = col3.number_input("Large", 100, 0, 1000)

ch4_intensity = st.number_input("CH4 Intensity", 1.5, 0.1, 10.0)

if st.button("🚀 RUN ANALYSIS", type="primary"):
    results, status, state_weights = calculate_lehs_production(
        sector, practice, states, small_farms, med_farms, large_farms, 
        ch4_intensity, adoption_rate/100
    )
    
    if status != "✅ SUCCESS":
        st.error(status)
    else:
        # EXECUTIVE METRICS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💚 CH4", f"{results['CH4_P50']:,.0f}", f"P10-P90: {results['CH4_P10']:,.0f}-{results['CH4_P90']:,.0f}")
        col2.metric("💰 Income", f"₹{results['Income_P50']:.1f} Cr")
        col3.metric("❤️ DALYs", f"{results['DALYs_P50']:.0f}")
        col4.metric("👥 Jobs", f"{results['Jobs_P50']:.0f} FTE")
        
        # STATE BREAKDOWN
        state_df = pd.DataFrame([
            {"State": s, "CH4": results['CH4_P50']*w, "Income": results['Income_P50']*w}
            for s, w in state_weights.items() if s in states
        ])
        st.subheader("📊 State Breakdown")
        st.dataframe(state_df.round(1))
        
        # DOWNLOAD
        csv = state_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "lehs-report.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("*BAHS 2023 | NDDB | IRRI | GBD-MAPS*")
