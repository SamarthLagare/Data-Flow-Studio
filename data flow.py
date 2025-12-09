import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from io import BytesIO

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DataFlow Auto",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State Management
if 'page' not in st.session_state: st.session_state.page = 'Home'
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Layout */
    .block-container { 
        max-width: 1200px; 
        padding-top: 2rem; 
        padding-bottom: 5rem; 
    }

    /* Card Styling */
    .nav-card {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    /* Primary Button */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        height: 3rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Auto-Pilot Button (Gradient) */
    .auto-pilot-btn > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        height: 4em !important;
        font-size: 1.2rem !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid #333;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #3b82f6; font-size: 1.5rem; }
    
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

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("⚡ DataFlow")
    st.markdown("Automated Intelligence")
    
    # NAVIGATION OPTIONS
    nav_options = ["Home", "⚡ Auto-Pilot", "1. Data Studio", "2. EDA Lab", "3. Model Forge"]
    
    # Sync Sidebar
    ix = 0
    if st.session_state.page == "Auto-Pilot": ix = 1
    elif st.session_state.page == "Data Studio": ix = 2
    elif st.session_state.page == "EDA Lab": ix = 3
    elif st.session_state.page == "Model Forge": ix = 4
    
    selected = st.radio("Navigate", nav_options, index=ix, label_visibility="collapsed")
    
    # Update Page
    if selected == "Home": st.session_state.page = "Home"
    elif selected == "⚡ Auto-Pilot": st.session_state.page = "Auto-Pilot"
    elif selected == "1. Data Studio": st.session_state.page = "Data Studio"
    elif selected == "2. EDA Lab": st.session_state.page = "EDA Lab"
    elif selected == "3. Model Forge": st.session_state.page = "Model Forge"

    st.markdown("---")
    st.caption("Project Stats")
    if st.session_state.df is not None:
        r, c = st.session_state.df.shape
        st.success(f"Data: {r} rows, {c} cols")
    else:
        st.info("No Data Loaded")

# --- 5. COMPONENT: AUTO-PILOT (NEW!) ---
def render_autopilot():
    st.header("⚡ Auto-Pilot")
    st.markdown("One click to Clean, Visualize, and Model your data.")
    
    # 1. LOAD
    if st.session_state.df is None:
        with st.container():
            uploaded_file = st.file_uploader("Upload CSV or Excel to Start", type=['csv', 'xlsx'])
            if uploaded_file:
                st.session_state.df = load_data(uploaded_file)
                st.rerun()
        return

    df = st.session_state.df
    
    # 2. CONFIG
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        st.markdown("### Configuration")
        target = st.selectbox("Target Variable (Prediction)", df.columns)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="auto-pilot-btn">', unsafe_allow_html=True)
        start_btn = st.button("🚀 START AUTO-PILOT")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. EXECUTION
    if start_btn:
        with st.spinner("🤖 Auto-Pilot Engaged..."):
            # --- PHASE 1: CLEANING ---
            clean_df = df.copy()
            clean_df = clean_df.drop_duplicates()
            
            # Impute
            num_cols = clean_df.select_dtypes(include=np.number).columns
            cat_cols = clean_df.select_dtypes(include='object').columns
            clean_df[num_cols] = clean_df[num_cols].fillna(clean_df[num_cols].median())
            clean_df[cat_cols] = clean_df[cat_cols].fillna("Unknown")
            
            # Encode for ML
            ml_df = clean_df.copy()
            le = LabelEncoder()
            for c in ml_df.select_dtypes(include='object').columns:
                ml_df[c] = le.fit_transform(ml_df[c].astype(str))
                
            # --- PHASE 2: MODELING ---
            is_class = len(ml_df[target].unique()) < 20
            X = ml_df.drop(columns=[target])
            y = ml_df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if is_class:
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
                metric_name = "Accuracy"
            else:
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                metric_name = "R² Score"

            # SAVE STATE
            st.session_state.df = clean_df
            st.session_state.model = model

    # 4. REPORT DASHBOARD (Only show if button pressed or if just ran)
    if 'model' in locals() or start_btn:
        st.divider()
        st.markdown("## 📑 Mission Report")
        
        # A. DATA HEALTH
        st.subheader("1. Data Health")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original Rows", len(df))
        m2.metric("Cleaned Rows", len(clean_df))
        m3.metric("Missing Fixed", df.isna().sum().sum())
        m4.metric("Duplicates Drop", df.duplicated().sum())
        
        # B. INSIGHTS
        st.subheader("2. Key Insights")
        t_eda1, t_eda2 = st.tabs(["Correlations", "Target Distribution"])
        
        with t_eda1:
            numeric_df = clean_df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix", template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True, key="auto_corr")
        
        with t_eda2:
            fig_hist = px.histogram(clean_df, x=target, title=f"Distribution of {target}", template="plotly_dark", color_discrete_sequence=['#3b82f6'])
            st.plotly_chart(fig_hist, use_container_width=True, key="auto_hist")

        # C. MODEL PERFORMANCE
        st.subheader("3. Predictive Power")
        c_res1, c_res2 = st.columns([1, 2])
        
        with c_res1:
            st.markdown(f"#### {metric_name}")
            st.markdown(f"<h1 style='color: #10b981; font-size: 3.5rem;'>{score:.2%}</h1>", unsafe_allow_html=True)
            st.caption(f"Model: Random Forest ({'Classifier' if is_class else 'Regressor'})")
            
        with c_res2:
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True).tail(10)
                fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', title="Top Drivers (Feature Importance)", template="plotly_dark")
                st.plotly_chart(fig_imp, use_container_width=True, key="auto_imp")

# --- 6. COMPONENT: DATA STUDIO ---
def render_data_studio():
    st.header("💿 Data Studio")
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.rerun()
    else:
        df = st.session_state.df
        t1, t2, t3 = st.tabs(["Overview", "Cleaning", "Transformation"])
        
        with t1:
            st.dataframe(df.head(), use_container_width=True)
            st.write(df.describe())
            
        with t2:
            col_clean = st.selectbox("Column to Clean", df.columns)
            if st.button("Fill NA with Median"):
                if pd.api.types.is_numeric_dtype(df[col_clean]):
                    st.session_state.df[col_clean] = df[col_clean].fillna(df[col_clean].median())
                    st.rerun()
            if st.button("Drop Duplicates"):
                st.session_state.df = df.drop_duplicates()
                st.rerun()

        with t3:
            filter_col = st.selectbox("Filter Column", df.columns)
            unique_val = st.selectbox("Value", df[filter_col].unique())
            if st.button("Apply Filter"):
                st.session_state.df = df[df[filter_col] == unique_val]
                st.rerun()

# --- 7. COMPONENT: EDA LAB ---
def render_eda():
    st.header("📊 EDA Lab")
    if st.session_state.df is None: st.warning("No Data"); return
    
    df = st.session_state.df
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("### Settings")
        chart = st.selectbox("Chart Type", ["Histogram", "Box", "Scatter", "Line", "Bar", "Area", "Violin", "Pie", "Heatmap", "3D Scatter"])
        x_ax = st.selectbox("X Axis", all_cols)
        y_ax = st.selectbox("Y Axis", num_cols if chart != "Pie" else all_cols)
        color = st.selectbox("Color", [None] + all_cols)
    
    with c2:
        if chart == "Histogram": fig = px.histogram(df, x=x_ax, color=color, template="plotly_dark")
        elif chart == "Box": fig = px.box(df, x=x_ax, y=y_ax, color=color, template="plotly_dark")
        elif chart == "Scatter": fig = px.scatter(df, x=x_ax, y=y_ax, color=color, template="plotly_dark")
        elif chart == "Line": fig = px.line(df, x=x_ax, y=y_ax, color=color, template="plotly_dark")
        elif chart == "Bar": fig = px.bar(df, x=x_ax, y=y_ax, color=color, template="plotly_dark")
        elif chart == "Area": fig = px.area(df, x=x_ax, y=y_ax, color=color, template="plotly_dark")
        elif chart == "Violin": fig = px.violin(df, x=x_ax, y=y_ax, color=color, template="plotly_dark")
        elif chart == "Pie": fig = px.pie(df, names=x_ax, template="plotly_dark")
        elif chart == "Heatmap": fig = px.density_heatmap(df, x=x_ax, y=y_ax, template="plotly_dark")
        elif chart == "3D Scatter": 
            z_ax = st.selectbox("Z Axis", num_cols)
            fig = px.scatter_3d(df, x=x_ax, y=y_ax, z=z_ax, color=color, template="plotly_dark")
            
        st.plotly_chart(fig, use_container_width=True, key="manual_eda")

# --- 8. COMPONENT: MODEL FORGE ---
def render_model():
    st.header("🧠 Model Forge")
    if st.session_state.df is None: st.warning("No Data"); return
    
    df = st.session_state.df.copy().dropna()
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c].astype(str))
        
    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.selectbox("Target", df.columns)
        feats = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        
        task = "Regression" if len(df[target].unique()) > 20 else "Classification"
        st.caption(f"Task: {task}")
        
        if task == "Regression":
            algo = st.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Ridge", "Lasso", "Gradient Boosting"])
        else:
            algo = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "SVC", "KNN", "Gradient Boosting"])
            
        if st.button("Train"):
            X, y = df[feats], df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            if task == "Regression":
                model = RandomForestRegressor() if algo == "Random Forest" else LinearRegression()
                model.fit(X_train, y_train)
                st.metric("R2 Score", f"{r2_score(y_test, model.predict(X_test)):.3f}")
            else:
                model = RandomForestClassifier() if algo == "Random Forest" else LogisticRegression()
                model.fit(X_train, y_train)
                st.metric("Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")
            
            st.session_state.model = model

    with c2:
        if st.session_state.model:
            st.success("Model Trained")
            # Feature Importance
            if hasattr(st.session_state.model, 'feature_importances_'):
                imp = pd.DataFrame({'Feature': feats, 'Importance': st.session_state.model.feature_importances_}).sort_values('Importance', ascending=True)
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h', template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True, key="man_imp")

# --- 9. PAGE ROUTING ---

if st.session_state.page == 'Home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("DataFlow")
    st.markdown("### Choose your workflow")
    
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Auto-Pilot")
        st.markdown("Let AI handle everything. Best for quick insights.")
        st.markdown('<div class="auto-pilot-btn">', unsafe_allow_html=True)
        if st.button("Start Auto-Pilot"): st.session_state.page = "Auto-Pilot"; st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Manual Mode")
        st.markdown("Granular control over Cleaning, EDA, and Modeling.")
        if st.button("Enter Studio"): st.session_state.page = "Data Studio"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Auto-Pilot': render_autopilot()
elif st.session_state.page == 'Data Studio': render_data_studio()
elif st.session_state.page == 'EDA Lab': render_eda()
elif st.session_state.page == 'Model Forge': render_model()
