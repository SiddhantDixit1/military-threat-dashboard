import streamlit as st
import pandas as pd
import pydeck as pdk
from data_pipeline import generate_intelligence_data
from model_engine import ThreatModelEngine
# Import our brand new UI components file
import ui_components 

# App configuration
st.set_page_config(page_title="Intel Threat Command Dashboard", layout="wide")

# 1. Apply our custom UI/UX styles
ui_components.apply_custom_theme()

# Title and Layout Header
st.title("🛡️ AI Military Intelligence Dashboard")
st.caption("Strategic Analytics, Multi-Modal Sensor Fusion, and Explainable AI (XAI) Prototype")
st.markdown("---")

# Initialize backend components
@st.cache_resource
def load_engine():
    return ThreatModelEngine()

engine = load_engine()

# Load live data pool
raw_data = generate_intelligence_data(50)

# 2. Render Sidebar using UI components
ui_components.render_sidebar_header()
severity_filter = st.sidebar.multiselect(
    "Filter by Threat Type:",
    options=list(raw_data['threat_type'].unique()),
    default=list(raw_data['threat_type'].unique())
)

# Filter Data
filtered_data = raw_data[raw_data['threat_type'].isin(severity_filter)]

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Active Inputs", len(filtered_data))
with col2:
    st.metric("Monitored Grid Sectors", "Sector 7-G")
with col3:
    st.metric("System Defense Status", "OPTIMAL", delta="100% Operational")

st.markdown("### 🗺️ Live Tactical Map Operations")

# Geospatial visualization layer using PyDeck
if not filtered_data.empty:
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v9',
        initial_view_state=pdk.ViewState(
            latitude=filtered_data['latitude'].mean(),
            longitude=filtered_data['longitude'].mean(),
            zoom=7,
            pitch=30,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_data,
                get_position='[longitude, latitude]',
                get_color='[220, 53, 69, 200]', 
                get_radius=4000,
                pickable=True
            ),
        ],
    ))
else:
    st.info("No active threats found for the selected filters.")

# Tactical Intelligence Data Feed
st.markdown("### 📋 Active Sensor Streams")
st.dataframe(filtered_data, use_container_width=True)

# Interactive Evaluation Section for AI Analysis
st.markdown("---")
st.markdown("### 🧠 Real-Time Explainable AI (XAI) Target Assessment")
st.write("Select an individual sensor stream ID row to see why the machine learning classifier flagged it.")

target_index = st.selectbox("Select Target Stream Index for Deep Diagnostics:", filtered_data.index)

if target_index is not None:
    selected_row = filtered_data.loc[[target_index]]
    analysis = engine.predict_threat(selected_row)
    
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        # Wrap this block inside our custom UI HTML card
        st.markdown(f"""
            <div class='tactical-card'>
                <h4>Classification Output</h4>
                <h3><b>{analysis['predicted_threat']}</b></h3>
            </div>
        """, unsafe_allow_html=True)
        
        score = analysis['risk_score']
        if score > 75:
            st.error(f"Calculated Threat Risk Score: {score}% (CRITICAL)")
        elif score > 40:
            st.warning(f"Calculated Threat Risk Score: {score}% (ELEVATED)")
        else:
            st.success(f"Calculated Threat Risk Score: {score}% (LOW RISK)")
            
    with diag_col2:
        st.subheader("🔍 Explainable AI (XAI) Metrics")
        st.markdown("**Key Risk Drivers Triggering This Assessment:**")
        for driver in analysis['top_drivers']:
            st.markdown(f"- 🔴 `{driver.replace('_', ' ').upper()}`")

# 3. Add footer
ui_components.render_footer()