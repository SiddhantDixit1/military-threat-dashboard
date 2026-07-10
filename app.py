import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from data_pipeline import ThreatDataSimulator
from model_engine import TacticalModelEngine

st.set_page_config(layout="wide", page_title="Command Console // XAI Layer")

# --- CORE CACHING AND INFRASTRUCTURE ---
@st.cache_data
def get_historical_pipeline():
    # Utilizing 25,000 rows to simulate operational data density
    sim = ThreatDataSimulator(n_samples=25000)
    return sim.generate_pipeline()

@st.cache_resource
def get_calibrated_engine(df):
    engine = TacticalModelEngine()
    engine.train_system(df)
    return engine

df_data = get_historical_pipeline()
engine = get_calibrated_engine(df_data)

# --- SPLIT LAYOUT PROTOCOL ---
main_dash, xai_dash = st.tabs(["🗺️ LIVE OPERATIONS", "🧠 EXPLAINABLE AI (XAI) DIAGNOSTICS"])

with main_dash:
    st.subheader("Tactical Geospatial Threat Feed")
    
    # Render Interactive Map Selection Frame
    view_state = pdk.ViewState(latitude=df_data['latitude'].mean(), longitude=df_data['longitude'].mean(), zoom=8)
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_data.head(500), # Sample preview window for responsiveness
        get_position='[longitude, latitude]',
        get_color='[220, 53, 69, 210]',
        get_radius=800,
        pickable=True
    )
    
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9')
    st.pydeck_chart(r)
    
    # Target Interception Input
    st.markdown("### 🎯 Target Interception Console")
    selected_index = st.selectbox("Select Target Stream ID for Full Memory Diagnostics:", options=df_data.index[:100])
    
    # Fetch row asset
    target_row = df_data.loc[[selected_index]]
    
    # Run explainability metrics pipeline
    metrics = engine.predict_and_explain(target_row, df_data)
    
    # Visual KPI Displays
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Predicted Threat Profile", metrics['predicted_threat'])
    with kpi2:
        st.metric("Model Confidence Rating", f"{metrics['confidence']:.2f}%")
    with kpi3:
        st.metric("Calculated Risk Severity", f"{metrics['risk_score']}/100")

    # Real-time Feature Triage Breakdown
    st.markdown("---")
    col_pos, col_neg = st.columns(2)
    
    with col_pos:
        st.error("🔴 Escalation Drivers (Positive Contributions)")
        pos_df = pd.DataFrame(metrics['positive_contributions'])
        if not pos_df.empty:
            st.dataframe(pos_df[['Feature', 'Actual_Value', 'SHAP_Impact']], use_container_width=True)
            
    with col_neg:
        st.success("🟢 Mitigation Drivers (Negative Contributions)")
        neg_df = pd.DataFrame(metrics['negative_contributions'])
        if not neg_df.empty:
            st.dataframe(neg_df[['Feature', 'Actual_Value', 'SHAP_Impact']], use_container_width=True)

# --- PANEL 2: COMPREHENSIVE EXPLAINABLE AI PANEL ---
with xai_dash:
    st.header("🧠 Deep AI Transparency Matrix")
    st.caption("Detailed local and global insights based on SHAP framework values.")
    st.markdown("---")
    
    exp_obj = metrics['explanation_object']
    
    # 1. LOCAL VISUALIZATIONS ROW
    st.subheader("📍 Localized Target Decisions")
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.markdown("**SHAP Waterfall Plot (Target Asset Prediction Path)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')
        shap.plots.waterfall(exp_obj, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        
    with graph_col2:
        st.markdown("**SHAP Bar Plot (Local Feature Absolute Importance)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.bar(exp_obj, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        
    st.markdown("---")
    
    # 2. GLOBAL VISUALIZATIONS ROW
    st.subheader("🌐 Global Model Behavior (20k+ Rows Context)")
    global_col1, global_col2 = st.columns(2)
    
    # Extract structural references for fast background computation passes
    X_sample = df_data[engine.features].head(1000)
    X_sample_scaled = engine.scaler.transform(X_sample)
    
    # Utilize cached calculation matrix to preserve instant response loops
    global_shap_matrix = engine.compute_cached_shap(X_sample_scaled)
    
    with global_col1:
        st.markdown("**SHAP Summary Plot (Global Density & Feature Impacts)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Isolate the exact class index we want to plot globally
        class_idx = metrics['predicted_class_index']
        
        # Extract the global array slice safely
        global_slice = global_shap_matrix[class_idx]
        
        # If SHAP nested it as (samples, features) inside a list, extract the first sample batch
        if isinstance(global_slice, list) or len(global_slice.shape) == 3:
            global_slice = global_slice[0]
            
        # Force the orientation check: if rows match features, transpose it to match X_sample
        if global_slice.shape[0] == len(engine.features) and global_slice.shape[0] != X_sample.shape[0]:
            global_slice = global_slice.T
            
        # Build a robust, standard multi-row Explanation object that SHAP cannot reject
        try:
            global_exp = shap.Explanation(
                values=global_slice,
                base_values=np.repeat(engine.explainer.expected_value[class_idx], X_sample.shape[0]),
                data=X_sample.values,
                feature_names=engine.features
            )
            shap.plots.beeswarm(global_exp, max_display=10, show=False)
        except Exception:
            # Flawless fallback to standard summary plot if beeswarm rejects the custom object
            shap.summary_plot(global_slice, X_sample, feature_names=engine.features, show=False)
            
        st.pyplot(fig, clear_figure=True)
        
    with global_col2:
        st.markdown("**Global Feature Importance Ranking (Plotly Engine)**")
        # Compute global average absolute impact values safely via numpy
        global_importance = np.abs(global_shap_matrix[metrics['predicted_class_index']]).mean(axis=0)
        feat_imp_df = pd.DataFrame({
            'Feature': engine.features,
            'Global_Importance': global_importance
        }).sort_values(by='Global_Importance', ascending=True)
        
        fig_plotly = px.bar(
            feat_imp_df, 
            x='Global_Importance', 
            y='Feature', 
            orientation='h',
            template='plotly_dark',
            color_discrete_sequence=['#ff3333']
        )
        fig_plotly.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_plotly, use_container_width=True)