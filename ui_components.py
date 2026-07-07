import streamlit as st

def apply_custom_theme():
    """Applies custom tactical CSS styles to the dashboard."""
    st.markdown("""
        <style>
            /* Main Background styling */
            .stApp {
                background-color: #0e1117;
            }
            /* Custom styling for headers */
            h1 {
                color: #ff3333 !important;
                font-family: 'Courier New', Courier, monospace;
                font-weight: bold;
                text-shadow: 0 0 10px rgba(255, 51, 51, 0.3);
            }
            /* Custom tactical card container */
            .tactical-card {
                padding: 20px;
                background-color: #1f2937;
                border-left: 5px solid #ff3333;
                border-radius: 4px;
                margin-bottom: 15px;
            }
            /* Metrics font tweak */
            [data-testid="stMetricValue"] {
                font-family: 'Courier New', Courier, monospace;
                font-size: 2rem;
                color: #00ff66 !important;
            }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar_header():
    """Renders a polished sidebar header."""
    # Using a reliable open-source icon URL with supported parameters
    st.sidebar.image("https://img.icons8.com/ios-filled/100/ffffff/radar.png", width=50)
    st.sidebar.markdown("# 🧠 COMMAND CONSOLE")
    st.sidebar.markdown("---")
def render_footer():
    """Renders a professional portfolio footer."""
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>⚠️ SECURE PROTOCOL // DEMO MODE ACTIVE</p>", 
        unsafe_allow_html=True
    )