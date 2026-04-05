# 🔒 Cybersecurity Intrusion Detection System

A comprehensive machine learning application for network intrusion detection using classification and regression models.

## 🚀 Features

- **Dual Model Support**: Classification (XGBoost) and Regression (Duration Prediction)
- **Real-time Predictions**: Interactive web interface for threat analysis
- **Model Comparison**: Test both approaches with the same dataset
- **Visualization**: Confusion matrices, learning curves, and feature analysis

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Models**: XGBoost, Random Forest
- **Data Processing**: scikit-learn, pandas, numpy
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📊 Dataset

Uses the IoT Network Intrusion Dataset with features for:
- Network traffic analysis
- Protocol detection
- Attack type classification
- Duration prediction

## 🚀 Deployment on Streamlit Cloud

### Quick Deploy

1. **Connect Repository**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `endalkmamo123-cell/machinelearningass1`

2. **Configure App**:
   - **Main file**: `app.py`
   - **Python version**: 3.9+ (recommended)
   - **Requirements file**: `requirements.txt`

3. **Deploy**:
   - Click "Deploy"
   - Wait for build completion (~5-10 minutes)

### Troubleshooting

#### Model Loading Issues
- Ensure all `.pkl` files are committed to the repository
- Check file paths are correct (case-sensitive on cloud)
- Verify `requirements.txt` includes all dependencies

#### Memory Issues
- Model files are optimized (<6MB total)
- App uses `@st.cache_resource` for efficient loading

#### Build Failures
- Check Streamlit Cloud logs for specific errors
- Ensure all imports are available in `requirements.txt`
- Test locally: `streamlit run app.py`

## 🏃‍♂️ Local Development

```bash
# Clone repository
git clone https://github.com/endalkmamo123-cell/machinelearningass1.git
cd machinelearningass1

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── CLASSIFICATION/          # Classification models and data
│   ├── best_xgb_model.pkl
│   ├── best_rf_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── *.csv               # Training data
└── REGRESSION/             # Regression models and data
    ├── regression_xgb_model.pkl
    ├── regression_encoders.pkl
    └── *.csv               # Training data
```

## 🎯 Usage

1. **Select Model**: Choose between "Standard Classification" or "Regression-Folder Model"
2. **Upload Data**: Use the file uploader or sample data
3. **Get Predictions**: View real-time threat analysis
4. **Compare Results**: Switch between models to compare performance

## 📈 Model Performance

### Classification (XGBoost)
- Accuracy: ~21%
- Best for: Multi-class attack type detection

### Regression (Duration Prediction)
- Predicts attack duration
- Useful for threat prioritization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to GitHub
5. Create a Pull Request

## 📄 License

This project is for educational and research purposes in cybersecurity and machine learning.

---

**Built with ❤️ for network security research**