import streamlit as st

st.set_page_config(
    page_title="Choose Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------- Styling ----------
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    margin-top: 80px;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 60px;
}

/* Navigation buttons */
div.stButton > button {
    width: 100%;
    height: 180px;
    border-radius: 15px;
    border: 1px solid #374151;
    background-color: #111827;
    color: white;
    font-size: 24px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #60a5fa;
    background-color: #1f2937;
    transform: translateY(-3px);
}

</style>
""", unsafe_allow_html=True)

# ---------- Header ----------

st.markdown(
    '<div class="main-title">Choose Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Pricing, simulation and numerical analysis</div>',
    unsafe_allow_html=True
)

# ---------- Navigation ----------
col1, col2 = st.columns(2, gap="large")

with col1:

    if st.button(
        "📈\n\nOptions Pricing",
        key="Pricing",
        use_container_width=True
    ):
        st.switch_page("pages/pricing.py")

with col2:

    if st.button(
        "📊\n\nMarket Modelling",
        key="Market Modelling",
        use_container_width=True
    ):
        st.switch_page("pages/dynamics.py")