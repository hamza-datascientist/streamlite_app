import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Predictor")
st.markdown("---")

# Demo Model (Apna model yahan load karo)
@st.cache_resource
def load_model():
    """Demo model banata hai"""
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    y = 1000000 + X[:,0]*50000 + X[:,1]*200000 + X[:,2]*150000 + X[:,3]*100000 - X[:,4]*50000
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# Sidebar mein inputs
st.sidebar.header("📝 Enter House Details")

area = st.sidebar.slider("📏 Area (sqft)", 500, 5000, 2000)
bedrooms = st.sidebar.slider("🛏️ Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("🚿 Bathrooms", 1, 4, 2)
location_score = st.sidebar.slider("📍 Location Score (1-10)", 1, 10, 7)
age = st.sidebar.slider("📅 House Age (years)", 0, 50, 10)

# Main area mein prediction
col1, col2 = st.columns([2,1])

with col1:
    st.header("🔮 Prediction")
    if st.button("🎯 Predict Price", type="primary", use_container_width=True):
        features = np.array([[area, bedrooms, bathrooms, location_score, age]])
        prediction = model.predict(features)[0]
        
        st.balloons()
        st.markdown(f"""
        <div style="background-color: #10B981; color: white; padding: 30px; 
                    border-radius: 15px; text-align: center; font-size: 28px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.2)">
            <h1>₹{prediction:,.0f}</h1>
            <p>Predicted House Price</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.header("📊 Summary")
    st.metric("Area", f"{area:,} sqft")
    st.metric("Bedrooms", bedrooms)
    st.metric("Bathrooms", bathrooms)
    st.metric("Location", f"{location_score}/10")

st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit*")
