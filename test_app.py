import streamlit as st
import pandas as pd
import numpy as np

st.title("🚀 Deployment Test")
st.success("✅ App loads successfully!")

# Test basic functionality
st.write("Testing basic imports:")
st.write("✅ streamlit")
st.write("✅ pandas")
st.write("✅ numpy")

# Test file access
import os
st.write("\\nTesting file access:")
script_dir = os.path.dirname(os.path.abspath(__file__))
st.write(f"Script directory: {script_dir}")

if os.path.exists("CLASSIFICATION"):
    st.write("✅ CLASSIFICATION folder found")
else:
    st.write("❌ CLASSIFICATION folder not found")

if os.path.exists("requirements.txt"):
    st.write("✅ requirements.txt found")
else:
    st.write("❌ requirements.txt not found")

st.balloons()