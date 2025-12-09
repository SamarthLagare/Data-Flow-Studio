import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from io import BytesIO

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DataFlow",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State Management
if 'page' not in st.session_state: st.session_state.page = 'Home'
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None

# --- 2. EASY-READ CSS ---
st.markdown("""
    <style>
    /* Global Font - Clean & Readable */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Layout Spacing */
    .block-container { 
        max-width: 1000px; 
        padding-top: 2rem; 
        padding-bottom: 5rem; 
    }

    /* Card Styling - High Contrast */
    .nav-card {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .nav-card:hover {
        border-color: #3b82f6;
    }

    /* Headings */
    h1, h2, h3 { color: white; font-weight: 600; }
    p { color: #cfcfcf; font-size: 1.05rem; }

    /* Primary Button - Big & Blue */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        height: 3.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid #333;
    }
    
    /* Navigation Radio Buttons */
    div[role="radiogroup"] label {
        font-size: 1.1rem !important;
        padding: 10px 5px;
        border-radius: 5px;
    }
    div[role="radiogroup"] label:hover {
        background-color: #1f2937;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] { color: #3b82f6; }
    
    /* Hide Junk */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def load_data(file):
    try:
        if file.name.endswith('.csv'): return pd.read_csv(file)
        elif file.name.endswith('.xlsx'): return pd.read_excel(file)
    except: return None

def navigate_to(page_name):
    st.session_state.page = page_name

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🌊 DataFlow")
    st.markdown("Automated Analytics")
    st.markdown("---")
    
    # 1. Main Navigation
    st.subheader("📍 Menu")
    nav_options = ["Home", "1. Data Studio", "2. EDA Lab", "3. Model Forge"]
    
    # Sync Sidebar with Session State
    current_index = 0
    if st.session_state.page == "Data Studio": current_index = 1
    elif st.session_state.page == "EDA Lab": current_index = 2
    elif st.session_state.page == "Model Forge": current_index = 3
    
    selected = st.radio("Navigate", nav_options, index=current_index, label_visibility="collapsed")
    
    # Update state based on selection
    if selected == "Home": st.session_state.page = "Home"
    elif selected == "1. Data Studio": st.session_state.page = "Data Studio"
    elif selected == "2. EDA Lab": st.session_state.page = "EDA Lab"
    elif selected == "3. Model Forge": st.session_state.page = "Model Forge"

    # 2. Project Status (Context Aware)
    st.markdown("---")
    st.subheader("📊 Project Status")
    if st.session_state.df is not None:
        rows, cols = st.session_state.df.shape
        st.success(f"Data Loaded: {rows} rows")
    else:
        st.info("No Data Loaded")
        
    if st.session_state.model is not None:
        st.success("Model Trained: Ready")

# --- 5. COMPONENT: DATA STUDIO ---
def render_data():
    st.header("💿 Data Studio")
    st.markdown("Upload your file and we will clean it for you.")
    
    # Upload
    with st.container():
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Two Columns: Stats & Actions
        c1, c2 = st.columns([1, 1], gap="large")
        
        with c1:
            st.subheader("Quick Stats")
            col1, col2 = st.columns(2)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            st.metric("Missing Values", df.isna().sum().sum())
            
        with c2:
            st.subheader("Automation")
            st.info("Click below to remove duplicates & fill missing values.")
            if st.button("✨ Auto-Clean Dataset"):
                df = df.drop_duplicates()
                num_cols = df.select_dtypes(include=np.number).columns
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                cat_cols = df.select_dtypes(include='object').columns
                df[cat_cols] = df[cat_cols].fillna("Unknown")
                st.session_state.df = df
                st.success("Cleaning Complete!")
                st.rerun()

        st.markdown("---")
        with st.expander("Show Data Preview"):
            st.dataframe(df, use_container_width=True)

# --- 6. COMPONENT: EDA LAB ---
def render_eda():
    st.header("📊 EDA Lab")
    
    if st.session_state.df is None:
        st.warning("Please go to Data Studio and upload a file first.")
        if st.button("Go to Data Studio"):
            navigate_to("Data Studio")
            st.rerun()
        return

    df = st.session_state.df
    
    # Simple Tabs for Visualization
    t1, t2 = st.tabs(["📈 Distributions", "🔗 Relationships"])
    
    with t1:
        st.caption("Understand your data distribution.")
        c1, c2 = st.columns([1, 3])
        with c1:
            col_name = st.selectbox("Select Column", df.columns, key="dist_c")
            color_by = st.selectbox("Color By (Optional)", [None] + list(df.columns), key="dist_hue")
        with c2:
            fig = px.histogram(df, x=col_name, color=color_by, template="plotly_dark", marginal="box")
            st.plotly_chart(fig, use_container_width=True, key="p_dist")

    with t2:
        st.caption("Compare two variables.")
        c1, c2 = st.columns([1, 3])
        with c1:
            x_ax = st.selectbox("X Axis", df.columns, key="rel_x")
            y_ax = st.selectbox("Y Axis", df.select_dtypes(include=np.number).columns, key="rel_y")
            hue = st.selectbox("Color", [None] + list(df.columns), key="rel_hue")
        with c2:
            fig = px.scatter(df, x=x_ax, y=y_ax, color=hue, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True, key="p_rel")

# --- 7. COMPONENT: MODEL FORGE ---
def render_model():
    st.header("🧠 Model Forge")
    
    if st.session_state.df is None:
        st.warning("Please go to Data Studio and upload a file first.")
        return

    df = st.session_state.df.copy()
    
    # Auto-Encode for ML
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c].astype(str))
        
    c1, c2 = st.columns([1, 2], gap="large")
    
    with c1:
        st.subheader("Setup")
        target = st.selectbox("Target Variable (What to predict?)", df.columns)
        
        # Logic to guess type
        if len(df[target].unique()) < 10:
            task = "Classification"
            st.info("Detected: Classification")
        else:
            task = "Regression"
            st.info("Detected: Regression")
            
        st.markdown("---")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training..."):
                X = df.drop(columns=[target])
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                if task == "Classification":
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    msg = f"Accuracy: {score:.1%}"
                else:
                    model = RandomForestRegressor()
                    model.fit(X_train, y_train)
                    score = r2_score(y_test, model.predict(X_test))
                    msg = f"R² Score: {score:.3f}"
                
                st.session_state.model_score = msg
                st.session_state.model = model
                st.toast("Training Done!", icon="🎉")

    with c2:
        st.subheader("Results")
        if st.session_state.model:
            st.markdown('<div class="nav-card">', unsafe_allow_html=True)
            st.metric("Model Performance", st.session_state.get("model_score", "N/A"))
            st.caption("Model trained using Random Forest algorithm.")
            
            # Download
            out = BytesIO()
            pickle.dump(st.session_state.model, out)
            st.download_button("⬇️ Download Model File", out.getvalue(), "model.pkl")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("*Click 'Train Model Now' to see results here.*")

# --- 8. PAGE ROUTING ---

if st.session_state.page == 'Home':
    st.title("🌊 DataFlow")
    st.markdown("### Simple. Automated. Analytics.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 1. Data Studio")
        st.caption("Upload & Auto-Clean")
        if st.button("Start Import"):
            navigate_to("Data Studio")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 2. EDA Lab")
        st.caption("Visualize Trends")
        if st.button("Start Analysis"):
            navigate_to("EDA Lab")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 3. Model Forge")
        st.caption("One-Click ML")
        if st.button("Start Prediction"):
            navigate_to("Model Forge")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Data Studio': render_data()
elif st.session_state.page == 'EDA Lab': render_eda()
elif st.session_state.page == 'Model Forge': render_model()
