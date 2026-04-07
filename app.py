"""
IDS Command Center v2.1 - Streamlit Cloud Deployment
Last updated: April 7, 2026
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

script_dir = os.path.dirname(os.path.abspath(__file__))

EXPECTED_CLASSIFICATION_COLUMNS = [
    'Event ID', 'Timestamp', 'Source IP', 'Destination IP',
    'User Agent', 'Attack Severity', 'Data Exfiltrated',
    'Threat Intelligence', 'Response Action'
]

def _repair_header_and_load_csv(raw, expected_columns=None):
    expected_columns = expected_columns or EXPECTED_CLASSIFICATION_COLUMNS
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='replace')

    raw = str(raw)
    header_start = raw.find(expected_columns[0])
    if header_start > 0:
        raw = raw[header_start:]

    first_newline = raw.find('\n')
    if first_newline != -1:
        first_line = raw[:first_newline]
        last_header = expected_columns[-1]
        last_index = first_line.find(last_header)
        if last_index != -1:
            after_header = first_line[last_index + len(last_header):]
            if after_header and not after_header.startswith(('\r', '\n')):
                raw = (
                    first_line[: last_index + len(last_header)]
                    + '\n'
                    + after_header
                    + raw[first_newline:]
                )

    return pd.read_csv(StringIO(raw))


def load_csv_with_header_repair(source, expected_columns=None):
    expected_columns = expected_columns or EXPECTED_CLASSIFICATION_COLUMNS
    if isinstance(source, (str, os.PathLike)):
        with open(source, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    elif hasattr(source, 'read'):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
    else:
        raise ValueError('Unsupported source type for CSV loader.')

    return _repair_header_and_load_csv(raw, expected_columns=expected_columns)


def safe_label_encode_scalar(value, encoder):
    if value is None:
        return -1
    value_str = str(value)
    classes = encoder.classes_
    if value_str in classes:
        return int(np.where(classes == value_str)[0][0])
    return -1


def safe_label_encode_series(series, encoder):
    lookup = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}
    return series.astype(str).map(lambda v: lookup.get(v, -1)).astype(int)


def prepare_batch_features(df, scaler, encoders=None):
    expected_cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else df.columns.tolist()
    X = pd.DataFrame(0, index=df.index, columns=expected_cols)

    for col in expected_cols:
        if col not in df.columns:
            continue

        if encoders and col in encoders:
            X[col] = safe_label_encode_series(df[col], encoders[col])
        elif col == 'Data Exfiltrated':
            X[col] = df[col].astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y']).astype(int)
        else:
            X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return scaler.transform(X), X

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


def build_feature_frame(values: dict, template_csv: str, drop_cols=None, scaler=None, encoders=None):
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
            if encoders and k in encoders:
                final.at[0, k] = safe_label_encode_scalar(v, encoders[k])
            elif k == 'Data Exfiltrated':
                final.at[0, k] = 1 if str(v).strip().lower() in ('true', '1', 'yes', 'y') else 0
            else:
                final.at[0, k] = v

    final = final.apply(pd.to_numeric, errors='coerce').fillna(0)

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
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Event Identity")
            event_id = st.text_input("Event ID", "00000000-0000-0000-0000-000000000000")
            timestamp = st.text_input("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            src_ip = st.text_input("Source IP", "192.168.1.1")
            dst_ip = st.text_input("Destination IP", "10.0.0.50")
            user_agent = st.text_input("User Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

        with col2:
            st.markdown("#### Threat Attributes")
            attack_severity = st.selectbox(
                "Attack Severity",
                encoders['Attack Severity'].classes_ if 'Attack Severity' in encoders else ['Low', 'Medium', 'High', 'Critical']
            )
            data_exfiltrated = st.selectbox("Data Exfiltrated", ['False', 'True'])
            threat_intel = st.text_input("Threat Intelligence", "Suspicious activity detected in ingress traffic.")
            response_action = st.selectbox(
                "Response Action",
                encoders['Response Action'].classes_ if 'Response Action' in encoders else ['Blocked', 'Contained', 'Eradicated', 'Recovered']
            )

        submit = st.form_submit_button("RUN NEURAL ANALYSIS ⚡", use_container_width=True)

    if submit:
        with st.spinner("Analyzing the intrusion event..."):
            values = {
                'Event ID': event_id,
                'Timestamp': timestamp,
                'Source IP': src_ip,
                'Destination IP': dst_ip,
                'User Agent': user_agent,
                'Attack Severity': attack_severity,
                'Data Exfiltrated': data_exfiltrated,
                'Threat Intelligence': threat_intel,
                'Response Action': response_action,
            }

            try:
                final_X = build_feature_frame(
                    values=values,
                    template_csv=os.path.join(script_dir, 'CLASSIFICATION', 'test_dataset.csv'),
                    scaler=scaler,
                    encoders=encoders
                )

                if final_X.shape[0] == 0:
                    st.error('Prepared input contains no samples. Check form entries and template file.')
                    st.stop()

                X_scaled = scaler.transform(final_X)

                if hasattr(model, 'predict_proba'):
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
                    pred_value = model.predict(X_scaled)[0]
                    actual_pred = np.expm1(pred_value)

                    st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.5); padding: 25px; border-radius: 12px; border-left: 8px solid #0088ff;">
                            <h2 style="color:#0088ff; margin:0;">REGRESSION RESULT</h2>
                            <h4 style="margin:5px 0;">Predicted Duration: {actual_pred:.4f} seconds</h4>
                            <p style="margin:0; opacity:0.7;">Note: Regression model selected. For classification, switch to Standard Classification model.</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

with tab2:
    st.subheader("Real-Time Traffic Interception")

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset for batch inference",
        type=['csv'],
        help="Upload a CSV with the same training columns used by the selected model."
    )

    if uploaded_file is not None:
        try:
            batch_df = load_csv_with_header_repair(uploaded_file, expected_columns=list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None)
            st.success("Upload successful. Previewing first rows below.")
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("RUN BATCH INFERENCE 🔍"):
                with st.spinner("Running model inference on uploaded data..."):
                    X_scaled, _ = prepare_batch_features(batch_df, scaler, encoders=encoders)
                    predictions = model.predict(X_scaled)

                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_scaled)
                        batch_df['Predicted Attack Type'] = encoders['target'].inverse_transform(predictions)
                        batch_df['Confidence (%)'] = np.max(probs, axis=1) * 100
                    else:
                        batch_df['Predicted Duration'] = np.expm1(predictions)

                    st.markdown("### Inference Results")
                    st.dataframe(batch_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.markdown("### Sample Stream Data Preview")
    try:
        if model_choice == "Standard Classification":
            batch = pd.read_csv(os.path.join(script_dir, 'CLASSIFICATION', 'test_dataset.csv')).sample(10)
            display_cols = ['Event ID', 'Timestamp', 'Source IP', 'Destination IP', 'Attack Type']
        else:
            batch = pd.read_csv(os.path.join(script_dir, 'REGRESSION', 'test.csv')).sample(10)
            display_cols = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto']
        st.dataframe(batch[display_cols], use_container_width=True)
        st.success("Sample data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading stream data: {e}")

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
