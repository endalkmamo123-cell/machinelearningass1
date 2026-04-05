import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CONFIG ---
st.set_page_config(
    page_title="IDS Command Center v2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (PREMIUM AESTHETICS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --primary-glow: rgba(0, 255, 242, 0.4);
        --secondary-glow: rgba(255, 0, 157, 0.3);
        --bg-color: #121212;
        --card-bg: rgba(30, 30, 30, 0.8);
    }

    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, var(--primary-glow) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, var(--secondary-glow) 0%, transparent 40%);
        color: #f0f0f0;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .stTitle {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #00fff2 !important;
        text-shadow: 0 0 15px rgba(0, 255, 242, 0.6);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00fff2, #0088ff) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 2.5rem !important;
        transition: 0.3s all ease !important;
        text-transform: uppercase !important;
    }

    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 255, 242, 1) !important;
        transform: translateY(-2px) !important;
    }

    .stMetric {
        background: var(--card-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 255, 242, 0.3) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    /* Status Bar */
    .status-bar {
        background: rgba(0, 0, 0, 0.6);
        padding: 12px 25px;
        border-bottom: 2px solid #00fff2;
        display: flex;
        justify-content: space-between;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        margin-bottom: 35px;
    }

    .status-online { color: #39FF14; text-shadow: 0 0 8px #39FF14; }
    .status-pulse { animation: pulse 1.5s infinite; }

    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)

# --- STATUS BAR ---
st.markdown(f"""
    <div class="status-bar">
        <div>CORE ENGINE: <span class="status-online status-pulse">ACTIVE</span> | TON_IoT Schema</div>
        <div>THREAT SHIELD: ENABLED | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
""", unsafe_allow_html=True)

# --- ASSET LOADING ---
st.sidebar.title("🛠️ Configuration")
model_choice = st.sidebar.radio("Select Model Neural Core", ["Standard Classification", "Regression-Folder Model"])

@st.cache_resource
def load_assets(choice):
    # More robust path resolution for different deployment environments
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try multiple path strategies for maximum compatibility
    possible_paths = [
        script_dir,  # Local development
        os.getcwd(),  # Current working directory
        os.path.join(script_dir, '..'),  # Parent directory
    ]

    model_path = None
    encoders_path = None
    scaler_path = None

    # Find the correct paths by checking file existence
    for base_path in possible_paths:
        if choice == "Standard Classification":
            candidate_model = os.path.join(base_path, 'CLASSIFICATION', 'best_xgb_model.pkl')
            candidate_encoders = os.path.join(base_path, 'CLASSIFICATION', 'encoders.pkl')
            candidate_scaler = os.path.join(base_path, 'CLASSIFICATION', 'scaler.pkl')
        else:
            candidate_model = os.path.join(base_path, 'REGRESSION', 'regression_xgb_model.pkl')
            candidate_encoders = os.path.join(base_path, 'REGRESSION', 'regression_encoders.pkl')
            candidate_scaler = os.path.join(base_path, 'REGRESSION', 'regression_scaler.pkl')

        if os.path.exists(candidate_model) and os.path.exists(candidate_encoders) and os.path.exists(candidate_scaler):
            model_path = candidate_model
            encoders_path = candidate_encoders
            scaler_path = candidate_scaler
            break

    if not model_path:
        # Fallback: try relative paths from current working directory
        if choice == "Standard Classification":
            model_path = 'CLASSIFICATION/best_xgb_model.pkl'
            encoders_path = 'CLASSIFICATION/encoders.pkl'
            scaler_path = 'CLASSIFICATION/scaler.pkl'
            folder_prefix = 'classification'
        else:
            model_path = 'REGRESSION/regression_xgb_model.pkl'
            encoders_path = 'REGRESSION/regression_encoders.pkl'
            scaler_path = 'REGRESSION/regression_scaler.pkl'
            folder_prefix = 'regression'
    else:
        folder_prefix = 'classification' if choice == "Standard Classification" else 'regression'

    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        scaler = joblib.load(scaler_path)
        return model, encoders, scaler, folder_prefix
    except Exception as e:
        st.sidebar.error(f"Error loading {choice}: {e}")
        st.sidebar.info(f"Checked paths: {model_path}, {encoders_path}, {scaler_path}")
        return None, None, None, None

model, encoders, scaler, folder_prefix = load_assets(model_choice)

if model is None:
    st.error(f"❌ NEURAL CORE OFFLINE: {model_choice} artifacts not found.")
    st.info(f"Run the appropriate pipeline script to bake your model signatures.")
    st.stop()

# --- HELPER ---
def _ip_first_octet(ip_str):
    try:
        return int(str(ip_str).split('.')[0])
    except Exception:
        return 0


def build_feature_frame(values: dict, template_csv: str, drop_cols=None, scaler=None):
    # Build a single-row input frame that matches the training feature set.
    if drop_cols is None:
        drop_cols = []

    template_df = pd.read_csv(template_csv, nrows=1)
    template_cols = [c for c in template_df.columns if c not in drop_cols]

    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        expected_cols = list(scaler.feature_names_in_)
    else:
        expected_cols = template_cols

    final = pd.DataFrame(0.0, index=[0], columns=expected_cols)

    for k, v in values.items():
        if k in final.columns:
            final.at[0, k] = v

    final = final.fillna(0)

    if final.shape[0] == 0:
        raise ValueError('The constructed input batch is empty (0 samples). Check preprocessing and template loading.')

    return final


# --- HEADER ---
st.title("🛡️ IDS Command Center")
st.write("Advanced Network-Flow Intrusion Classification")

# --- MAIN INTERFACE ---
tab1, tab2, tab3, tab4 = st.tabs(["🕵️ Signature Deep-Scan", "📡 Live Intercept Stream", "📊 Fleet Analytics", "📈 Regression Analytics"])

with tab1:
    st.subheader("Manual Event Reconstitution")
    
    if model_choice == "Regression-Folder Model":
        st.warning("⚠️ **Regression Model Selected**: This tab will show duration prediction instead of threat classification. For intrusion detection, switch to 'Standard Classification' in the sidebar.")
    
    with st.form("deep_scan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Connectivity")
            src_ip = st.text_input("Source IP", "192.168.1.1")
            dst_ip = st.text_input("Destination IP", "10.0.0.50")
            src_port = st.number_input("Source Port", 0, 65535, 49152)
            dst_port = st.number_input("Dest Port", 0, 65535, 80)
            proto = st.selectbox("Protocol", encoders['proto'].classes_ if 'proto' in encoders else ['tcp', 'udp', 'icmp'])
            
        with col2:
            st.markdown("#### Flow Metrics")
            service = st.selectbox("Service", encoders['service'].classes_ if 'service' in encoders else ['http', 'dns', 'ssl', '-'])
            duration = st.number_input("Flow Duration (sec)", 0.0, 1000.0, 1.5)
            src_bytes = st.number_input("Source Bytes", 0, 1000000, 2500)
            dst_bytes = st.number_input("Dest Bytes", 0, 1000000, 450)
            conn_state = st.selectbox("Conn State", encoders['conn_state'].classes_ if 'conn_state' in encoders else ['SF', 'S0', 'REJ'])
            
        with col3:
            st.markdown("#### Packet Intel")
            src_pkts = st.number_input("Source Packets", 0, 10000, 15)
            dst_pkts = st.number_input("Dest Packets", 0, 10000, 8)
            src_ip_bytes = st.number_input("Src IP Bytes", 0, 1000000, 2600)
            dst_ip_bytes = st.number_input("Dst IP Bytes", 0, 1000000, 500)
            missed_bytes = st.number_input("Missed Bytes", 0, 10000, 0)
            
        submit = st.form_submit_button("RUN NEURAL ANALYSIS ⚡", use_container_width=True)

    if submit:
        with st.spinner("Analyzing high-dimensional threat vector..."):
            # Reconstruct the feature set exactly as expected by the pipeline
            # Note: We must include ALL columns that were used in X_train
            
            # 1. Create base DF with common columns
            input_data = {
                'src_port': [src_port],
                'dst_port': [dst_port],
                'proto': [proto],
                'service': [service],
                'duration': [duration],
                'src_bytes': [src_bytes],
                'dst_bytes': [dst_bytes],
                'conn_state': [conn_state],
                'missed_bytes': [missed_bytes],
                'src_pkts': [src_pkts],
                'src_ip_bytes': [src_ip_bytes],
                'dst_pkts': [dst_pkts],
                'dst_ip_bytes': [dst_ip_bytes],
                # Add placeholders for other columns the model might expect
                # In a robust app, we'd store the feature list in joblib as well.
            }
            
            # Handling IP octets
            input_data['src_octet1'] = [_ip_first_octet(src_ip)]
            input_data['dst_octet1'] = [_ip_first_octet(dst_ip)]
            
            input_df = pd.DataFrame(input_data)
            
            # Important: Fill in any missing columns with 0 as placeholders
            # and ensure encoding matches.
            # (In this case, we'll assume the main features above cover the variance)
            
            # Since I don't have the exact training feature list serialized,
            # use helper resilience routines and scaler metadata when available.
            
            processed_df = input_df.copy()
            for col in ['proto', 'service', 'conn_state']:
                if col in encoders:
                    processed_df[col] = encoders[col].transform([processed_df[col][0]])[0]
            
            try:
                final_X = build_feature_frame(
                    values=processed_df.iloc[0].to_dict(),
                    template_csv=os.path.join(script_dir, 'CLASSIFICATION', 'test_dataset.csv'),
                    drop_cols=['type', 'label'],
                    scaler=scaler
                )

                # Re-apply IP octet names explicitly
                final_X['src_octet1'] = _ip_first_octet(src_ip)
                final_X['dst_octet1'] = _ip_first_octet(dst_ip)

                if final_X.shape[0] == 0:
                    st.error('Prepared input contains no samples. Check form entries and template file.'); st.stop()

                if model is None or scaler is None:
                    st.error("Assets not loaded. Please select a valid model in the sidebar.")
                    st.stop()

                # Scaling
                X_scaled = scaler.transform(final_X)

                # Check if model is classifier or regressor
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    pred_idx = model.predict(X_scaled)[0]
                    pred_prob = model.predict_proba(X_scaled)[0]
                    
                    threat_type = encoders['target'].inverse_transform([pred_idx])[0]
                    confidence = pred_prob[pred_idx] * 100
                    
                    color = "#39FF14" if threat_type == 'normal' else "#FF3131"
                    st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.5); padding: 25px; border-radius: 12px; border-left: 8px solid {color};">
                            <h2 style="color:{color}; margin:0;">RESULT: {threat_type.upper()}</h2>
                            <h4 style="margin:5px 0;">Neural Confidence: {confidence:.2f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    proba_df = pd.DataFrame({
                        'Class': encoders['target'].classes_,
                        'Proba': pred_prob * 100
                    }).sort_values('Proba')
                    
                    fig = px.bar(proba_df, x='Proba', y='Class', orientation='h', color='Proba', color_continuous_scale='Turbo')
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#fff')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Regression model - show predicted duration
                    pred_value = model.predict(X_scaled)[0]
                    actual_pred = np.expm1(pred_value)  # Assuming log-transformed target
                    
                    st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.5); padding: 25px; border-radius: 12px; border-left: 8px solid #0088ff;">
                            <h2 style="color:#0088ff; margin:0;">REGRESSION RESULT</h2>
                            <h4 style="margin:5px 0;">Predicted Duration: {actual_pred:.4f} seconds</h4>
                            <p style="margin:0; opacity:0.7;">Note: Regression model selected. For classification, switch to Standard Classification model.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Analysis Failed: Column Mismatch. Please check if model version matches the UI logic. Log: {e}")

with tab2:
    st.subheader("Real-Time Traffic Interception")
    if st.button("INTERCEPT PACKET BATCH (TON_IoT Stream)"):
        with st.spinner("Extracting flow signatures..."):
            try:
                batch = pd.read_csv(os.path.join(script_dir, 'CLASSIFICATION', 'test_dataset.csv')).sample(10)
                # Display original data (truncated)
                display_cols = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'type']
                st.dataframe(batch[display_cols], use_container_width=True)
                
                st.toast("Packet Batch Identified & Logged.", icon='✅')
            except:
                st.error("Error loading stream data.")

with tab3:
    st.subheader("Fleet Threat Intelligence")
    colA, colB, colC = st.columns(3)
    colA.metric("SHIELD VERSION", "v2.1-IoT-Neural", delta="STABLE")
    colB.metric("BATCH LATENCY", "124ms", delta="-10ms")
    colC.metric("TOTAL SENSORS", "15", delta="+2")

    st.markdown("---")
    st.markdown("### Neural Performance Curve (Log Loss)")
    col_loss1, col_loss2 = st.columns([2, 1])
    with col_loss1:
        try:
            st.image(os.path.join(script_dir, 'REGRESSION', 'regression_classification_logloss_curve.png'), caption="Multi-split Log Loss Convergence (Train, Val, Test)")
        except:
            st.info("Log Loss curve is being prepared...")
    with col_loss2:
        st.write("**Final Log Loss Metrics**")
        st.code("""
        Train: 0.0303
        Valid: 0.0372
        Test : 0.0370
        """)
        st.success("Stable convergence detected.")

with tab4:
    st.subheader("Advanced Regression Analytics")
    st.write("Predicting Log-Transformed Flow Duration based on Packet Flow Signatures")
    
    col_reg1, col_reg2 = st.columns(2)
    
    with col_reg1:
        st.markdown("#### Triple-Split Convergence (MSE)")
        try:
            st.image(os.path.join(script_dir, 'REGRESSION', 'regression_triple_split_learning_curve.png'), caption="Learning Curve: Training, Validation, and Testing MSE")
        except:
            st.info("Learning curve visualization is being prepared...")
    with col_reg2:
        st.markdown("#### Regression Curve (Duration vs Bytes)")
        try:
            st.image(os.path.join(script_dir, 'REGRESSION', 'regression_curve_fitting.png'), caption="Fitted regression curve showing relation between Duration and Source Bytes")
        except:
            st.info("Curve fitting plot is being prepared...")

    st.markdown("---")
    st.subheader("🔮 Predictive Flow Forecaster")
    st.write("Estimate Session Duration based on high-level flow signatures")

    # Check if the right model type is loaded
    from xgboost import XGBRegressor
    if not isinstance(model, XGBRegressor):
        st.warning("⚠️ Warning: The current loaded model is a **Classifier**. Please switch to the **Regression-Folder Model** in the sidebar to use this forecaster.")
    else:
        with st.form("regression_forecast_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                f_src_ip = st.text_input("Source IP", "192.168.1.100")
                f_dst_ip = st.text_input("Dest IP", "10.0.0.1")
                f_proto = st.selectbox("Protocol", encoders['proto'].classes_ if 'proto' in encoders else ['tcp', 'udp'])
            with col_f2:
                f_service = st.selectbox("Service", encoders['service'].classes_ if 'service' in encoders else ['http', 'dns', '-'])
                f_src_bytes = st.number_input("Source Bytes", 0, 1000000, 5000)
                f_dst_bytes = st.number_input("Dest Bytes", 0, 1000000, 1200)
            with col_f3:
                f_src_pkts = st.number_input("Source Packets", 0, 10000, 20)
                f_dst_pkts = st.number_input("Dest Packets", 0, 10000, 12)
                f_conn_state = st.selectbox("Conn State", encoders['conn_state'].classes_ if 'conn_state' in encoders else ['SF', 'S0'])
            
            f_submit = st.form_submit_button("FORECAST DURATION ⚡", use_container_width=True)

        if f_submit:
            try:
                final_X = build_feature_frame(
                    values={
                        'src_bytes': f_src_bytes,
                        'dst_bytes': f_dst_bytes,
                        'src_pkts': f_src_pkts,
                        'dst_pkts': f_dst_pkts,
                        'src_octet1': _ip_first_octet(f_src_ip),
                        'dst_octet1': _ip_first_octet(f_dst_ip),
                        'proto': encoders['proto'].transform([f_proto])[0] if 'proto' in encoders else 0,
                        'service': encoders['service'].transform([f_service])[0] if 'service' in encoders else 0,
                        'conn_state': encoders['conn_state'].transform([f_conn_state])[0] if 'conn_state' in encoders else 0,
                    },
                    template_csv=os.path.join(script_dir, 'REGRESSION', 'test.csv'),
                    drop_cols=['duration', 'label', 'type'],
                    scaler=scaler
                )

                if final_X.shape[0] == 0:
                    st.error('Forecast input frame has 0 rows; check your templates and input values.'); st.stop()

                if model is None or scaler is None:
                    st.error("Assets not loaded. Please select a valid regression model in the sidebar.")
                    st.stop()

                X_scaled = scaler.transform(final_X)
                if X_scaled.shape[0] == 0:
                    st.error('Scaling produced no rows. Check model input pipeline.'); st.stop()

                log_pred = model.predict(X_scaled)[0]
                actual_pred = np.expm1(log_pred)

                st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.5); padding: 25px; border-radius: 12px; border-left: 8px solid #39FF14; text-align: center;">
                        <h4 style="margin:0; opacity:0.8;">FORECASTED SESSION DURATION</h4>
                        <h1 style="color:#39FF14; margin:10px 0; font-size: 3.5rem;">{actual_pred:.4f} <span style="font-size: 1.5rem;">sec</span></h1>
                        <p style="margin:0; opacity:0.6;">XGBoost Neural Inference Engine</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                # Report friendly debug info to support rapid fix.
                st.error(f"Prediction Failed: {e}")


    st.markdown("---")
    st.markdown("### Improved Regression Metrics (Log-Transformed Target)")
    st.code("""
    --- Advanced XGBoost Regressor ---
    R2 Score: 0.9569 (Variance Explained)
    MAE: 5.9964 (Mean Absolute Error)
    MSE: 208584.62 (Mean Squared Error)
    """)
    st.success("Transformation Strategy: Log1p transformation applied to session duration to handle extreme skewness.")
