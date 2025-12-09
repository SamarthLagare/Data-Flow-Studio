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
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DataFlow",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# State Management
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None

# --- 2. CLEAN & USABLE CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0f1116;
        color: #e6e6e6;
    }
    
    /* Layout */
    .block-container { max-width: 1000px; padding-top: 3rem; }

    /* Cards */
    .glass-card {
        background-color: #1c1f26;
        border: 1px solid #2d313a;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Headings */
    h1, h2, h3 { font-weight: 700; color: white; letter-spacing: -0.5px; }
    p { color: #8b949e; line-height: 1.6; }

    /* Primary Button (Blue Gradient) */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        height: 3.5rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Secondary/Ghost Button */
    .ghost-btn > button {
        background: transparent;
        border: 1px solid #3b82f6;
        color: #3b82f6;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1c1f26;
        padding: 5px;
        border-radius: 12px;
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white !important;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] { color: #3b82f6; font-size: 2rem; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPERS ---
def load_data(file):
    try:
        if file.name.endswith('.csv'): return pd.read_csv(file)
        elif file.name.endswith('.xlsx'): return pd.read_excel(file)
    except: return None

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def download_model(model):
    out = BytesIO()
    pickle.dump(model, out)
    return out.getvalue()

# --- 4. COMPONENT: DATA STUDIO ---
def render_data():
    st.markdown("## 💿 Data Studio")
    st.markdown("Upload your data and let us handle the messy parts.")
    
    # Upload Section
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop CSV or Excel file here", type=['csv', 'xlsx'])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.success("Data loaded successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Quick Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", df.isna().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())
        
        st.markdown("### ⚡ Actions")
        
        col_a, col_b = st.columns([1, 1], gap="medium")
        
        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### **One-Click Clean**")
            st.caption("Automatically remove duplicates and fill missing values.")
            
            if st.button("✨ Auto-Clean Data"):
                # 1. Drop Duplicates
                df = df.drop_duplicates()
                # 2. Fill Numbers with Median
                num_cols = df.select_dtypes(include=np.number).columns
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                # 3. Fill Text with "Unknown"
                cat_cols = df.select_dtypes(include='object').columns
                df[cat_cols] = df[cat_cols].fillna("Unknown")
                
                st.session_state.df = df
                st.toast("Data cleaned automatically!", icon="✨")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### **Export**")
            st.caption("Download your processed dataset.")
            csv = convert_df(df)
            st.download_button("⬇️ Download CSV", csv, "clean_data.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("👀 View Raw Data"):
            st.dataframe(df, use_container_width=True)

# --- 5. COMPONENT: EDA LAB ---
def render_eda():
    st.markdown("## 📊 EDA Lab")
    
    if st.session_state.df is None:
        st.info("Please upload data in the Studio first.")
        return

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    t1, t2, t3 = st.tabs(["Distribution", "Relationships", "Correlation"])
    
    with t1:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("### Settings")
            dist_col = st.selectbox("Column", df.columns, key="dist_sel")
            dist_hue = st.selectbox("Color By", [None] + cat_cols, key="dist_hue")
        with c2:
            fig = px.histogram(df, x=dist_col, color=dist_hue, marginal="box", template="plotly_dark")
            # FIXED: Unique Key
            st.plotly_chart(fig, use_container_width=True, key="chart_dist")

    with t2:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("### Settings")
            rel_x = st.selectbox("X Axis", df.columns, key="rel_x")
            rel_y = st.selectbox("Y Axis", num_cols, key="rel_y")
            rel_hue = st.selectbox("Color", [None] + cat_cols, key="rel_hue")
        with c2:
            fig = px.scatter(df, x=rel_x, y=rel_y, color=rel_hue, template="plotly_dark")
            # FIXED: Unique Key
            st.plotly_chart(fig, use_container_width=True, key="chart_rel")

    with t3:
        if len(num_cols) > 1:
            fig = px.imshow(df[num_cols].corr(), text_auto=True, template="plotly_dark", color_continuous_scale="RdBu_r")
            # FIXED: Unique Key
            st.plotly_chart(fig, use_container_width=True, key="chart_corr")
        else:
            st.warning("Need numeric data for correlation.")

# --- 6. COMPONENT: MODEL FORGE ---
def render_model():
    st.markdown("## 🧠 Model Forge")
    
    if st.session_state.df is None:
        st.info("Please upload data in the Studio first.")
        return

    # Auto-Prep Data
    df_ml = st.session_state.df.copy().dropna()
    le = LabelEncoder()
    for col in df_ml.select_dtypes(include='object').columns:
        df_ml[col] = le.fit_transform(df_ml[col])

    # Layout
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 1. Target")
        target = st.selectbox("What do you want to predict?", df_ml.columns)
        
        # Auto-detect task type
        if len(df_ml[target].unique()) < 10:
            task_type = "Classification"
            st.caption("Detected: **Classification** (Categories)")
        else:
            task_type = "Regression"
            st.caption("Detected: **Regression** (Numbers)")
            
        st.markdown("### 2. Model")
        if task_type == "Classification":
            algo = st.selectbox("Algorithm", ["Random Forest", "Logistic Regression"])
        else:
            algo = st.selectbox("Algorithm", ["Random Forest", "Linear Regression"])
            
        # Advanced Toggle
        use_adv = st.toggle("Advanced Settings")
        params = {}
        if use_adv and "Random Forest" in algo:
            params['n'] = st.slider("Trees", 10, 200, 100)
        
        st.divider()
        train_btn = st.button("🚀 Train Model")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if train_btn:
            with st.spinner("Training model..."):
                X = df_ml.drop(columns=[target])
                y = df_ml[target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Model Logic
                if task_type == "Classification":
                    if algo == "Random Forest": model = RandomForestClassifier(n_estimators=params.get('n', 100))
                    else: model = LogisticRegression()
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.success(f"Training Complete! Accuracy: {acc:.1%}")
                    
                    # Confusion Matrix Chart
                    cm = confusion_matrix(y_test, preds)
                    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", template="plotly_dark")
                    # FIXED: Unique Key
                    st.plotly_chart(fig, use_container_width=True, key="chart_cm")
                    st.markdown('</div>', unsafe_allow_html=True)

                else: # Regression
                    if algo == "Random Forest": model = RandomForestRegressor(n_estimators=params.get('n', 100))
                    else: model = LinearRegression()
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.success(f"Training Complete! R² Score: {r2:.3f}")
                    
                    # Pred vs Actual Chart
                    fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted", template="plotly_dark")
                    fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
                    # FIXED: Unique Key
                    st.plotly_chart(fig, use_container_width=True, key="chart_reg")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Save
                st.session_state.model = model
                b_model = download_model(model)
                st.download_button("Download Model (.pkl)", b_model, "model.pkl")

# ==========================================
# 7. MAIN NAVIGATION
# ==========================================

# Sidebar
with st.sidebar:
    st.header("Navigation")
    
    # Map friendly names to session IDs
    nav_options = {"Home": "home", "Data Studio": "data", "EDA Lab": "eda", "Model Forge": "model"}
    
    # Reverse lookup for default index
    current_key = next((k for k, v in nav_options.items() if v == st.session_state.page), "Home")
    
    selected = st.radio("Go to", list(nav_options.keys()), index=list(nav_options.keys()).index(current_key))
    
    if nav_options[selected] != st.session_state.page:
        st.session_state.page = nav_options[selected]
        st.rerun()

# --- ROUTING ---
if st.session_state.page == 'home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("DataFlow")
    st.markdown("### Automated Analytics Platform")
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💿 Data")
        st.caption("Clean & Prepare")
        if st.button("Start Studio", key="h_d"):
            st.session_state.page = 'data'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Visualize")
        st.caption("Explore Trends")
        if st.button("Start EDA", key="h_e"):
            st.session_state.page = 'eda'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Predict")
        st.caption("Train Models")
        if st.button("Start AI", key="h_m"):
            st.session_state.page = 'model'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'data': render_data()
elif st.session_state.page == 'eda': render_eda()
elif st.session_state.page == 'model': render_model()
