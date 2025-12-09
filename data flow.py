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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DataFlow Intelligent",
    page_icon="🌊",
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
    
    /* Stats Card */
    .stat-card {
        background-color: #13161c;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
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
    
    /* Auto-Pilot Button */
    .auto-btn > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        height: 4em !important;
        font-size: 1.2rem !important;
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
    st.title("🌊 DataFlow X")
    st.markdown("Intelligent Analytics")
    
    nav_options = ["Home", "1. Auto-Pilot", "2. Data Studio", "3. EDA Lab", "4. Model Forge"]
    
    ix = 0
    if st.session_state.page == "Auto-Pilot": ix = 1
    elif st.session_state.page == "Data Studio": ix = 2
    elif st.session_state.page == "EDA Lab": ix = 3
    elif st.session_state.page == "Model Forge": ix = 4
    
    selected = st.radio("Navigate", nav_options, index=ix, label_visibility="collapsed")
    
    if selected == "Home": st.session_state.page = "Home"
    elif selected == "1. Auto-Pilot": st.session_state.page = "Auto-Pilot"
    elif selected == "2. Data Studio": st.session_state.page = "Data Studio"
    elif selected == "3. EDA Lab": st.session_state.page = "EDA Lab"
    elif selected == "4. Model Forge": st.session_state.page = "Model Forge"

    st.markdown("---")
    if st.session_state.df is not None:
        st.success("Dataset Loaded")
    else:
        st.info("No Data Loaded")

# --- 5. COMPONENT: AUTO-PILOT (SMART) ---
def render_autopilot():
    st.header("⚡ Smart Auto-Pilot")
    st.markdown("This tool cleans data and runs a **Model Tournament** to find the best algorithm.")
    
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
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown("### Target Selection")
        target = st.selectbox("What do you want to predict?", df.columns)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="auto-btn">', unsafe_allow_html=True)
        start_btn = st.button("🚀 FIND BEST MODEL")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. INTELLIGENT EXECUTION
    if start_btn:
        with st.status("🤖 Auto-Pilot Running...", expanded=True) as status:
            # --- PHASE 1: CLEANING ---
            status.write("🧹 Cleaning Data & Handling Missing Values...")
            clean_df = df.copy().drop_duplicates()
            
            # Numeric Imputation
            num_cols = clean_df.select_dtypes(include=np.number).columns
            clean_df[num_cols] = clean_df[num_cols].fillna(clean_df[num_cols].median())
            
            # Categorical Imputation
            cat_cols = clean_df.select_dtypes(include='object').columns
            clean_df[cat_cols] = clean_df[cat_cols].fillna("Unknown")
            
            # Encode
            status.write("🔢 Encoding Categorical Variables...")
            ml_df = clean_df.copy()
            le = LabelEncoder()
            for c in ml_df.select_dtypes(include='object').columns:
                ml_df[c] = le.fit_transform(ml_df[c].astype(str))
                
            # --- PHASE 2: TOURNAMENT ---
            status.write("⚔️ Starting Model Tournament...")
            
            is_class = len(ml_df[target].unique()) < 20
            X = ml_df.drop(columns=[target])
            y = ml_df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            best_score = -np.inf
            best_model = None
            best_name = ""
            results_log = {}

            # Define Candidates
            if is_class:
                candidates = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier()
                }
                metric_name = "Accuracy"
            else:
                candidates = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(n_estimators=100),
                    "Gradient Boosting": GradientBoostingRegressor()
                }
                metric_name = "R² Score"

            # Train Loop
            for name, model in candidates.items():
                status.write(f"Testing {name}...")
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                if is_class: score = accuracy_score(y_test, preds)
                else: score = r2_score(y_test, preds)
                
                results_log[name] = score
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            
            status.update(label="Mission Complete!", state="complete")
            st.session_state.df = clean_df
            
            # --- RESULTS ---
            st.divider()
            c_res1, c_res2 = st.columns([1, 1], gap="large")
            
            with c_res1:
                st.markdown("### 🏆 Winner")
                st.success(f"**{best_name}** is the best fit for your data.")
                st.metric(f"Best {metric_name}", f"{best_score:.2%}" if is_class else f"{best_score:.3f}")
                
            with c_res2:
                st.markdown("### 📊 Tournament Standings")
                # Create DataFrame for Chart
                res_df = pd.DataFrame(list(results_log.items()), columns=['Model', 'Score'])
                res_df = res_df.sort_values(by='Score', ascending=True)
                
                fig = px.bar(res_df, x='Score', y='Model', orientation='h', 
                             title=f"Model Comparison ({metric_name})", template="plotly_dark",
                             color='Score', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

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
    
    st.markdown('<div class="nav-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Chart Control**")
        chart = st.selectbox("Type", ["Histogram", "Box", "Scatter", "Line", "Bar", "Area", "Violin", "Pie", "Heatmap", "3D Scatter"])
    with c2:
        st.markdown("**Axes**")
        ac1, ac2, ac3 = st.columns(3)
        x_ax = ac1.selectbox("X", all_cols)
        y_ax = ac2.selectbox("Y", num_cols if chart != "Pie" else all_cols)
        color = ac3.selectbox("Color", [None] + all_cols)
    st.markdown('</div>', unsafe_allow_html=True)
    
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
        
    st.plotly_chart(fig, use_container_width=True)

# --- 8. COMPONENT: MODEL FORGE (IMPROVED UI) ---
def render_model():
    st.header("🧠 Model Forge")
    
    if st.session_state.df is None:
        st.info("Upload data in Data Studio first.")
        return

    df = st.session_state.df.copy().dropna()
    
    # Stats Header
    st.markdown("#### Data Health Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"<div class='stat-card'><b>Rows</b><br>{df.shape[0]}</div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='stat-card'><b>Columns</b><br>{df.shape[1]}</div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='stat-card'><b>Numeric Features</b><br>{len(df.select_dtypes(include=np.number).columns)}</div>", unsafe_allow_html=True)
    s4.markdown(f"<div class='stat-card'><b>Categorical Features</b><br>{len(df.select_dtypes(include='object').columns)}</div>", unsafe_allow_html=True)
    
    st.divider()

    # Preprocessing
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c].astype(str))
        
    c1, c2 = st.columns([1, 2], gap="large")
    
    with c1:
        st.subheader("1. Design")
        target = st.selectbox("Target Variable", df.columns)
        features = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        
        # Detect Type
        task = "Regression"
        if len(df[target].unique()) < 15: task = "Classification"
        
        st.subheader("2. Configure")
        if task == "Regression":
            algo = st.selectbox("Model", ["Linear Regression", "Random Forest", "Ridge", "Lasso", "Gradient Boosting", "SVR", "Decision Tree"])
        else:
            algo = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVC", "KNN", "Decision Tree"])
            
        params = {}
        with st.expander("Advanced Hyperparameters"):
            split = st.slider("Train/Test Split", 0.1, 0.5, 0.2)
            if "Random Forest" in algo or "Gradient" in algo:
                params['n'] = st.slider("Trees (Estimators)", 10, 500, 100)
            if "KNN" in algo:
                params['k'] = st.slider("Neighbors (K)", 1, 20, 5)
            if "SVC" in algo or "SVR" in algo:
                params['C'] = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
                
        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🚀 Train Model")

    with c2:
        st.subheader("3. Performance")
        
        if train_btn:
            with st.spinner("Training..."):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Model Selection
                model = None
                if task == "Regression":
                    if algo == "Linear Regression": model = LinearRegression()
                    elif algo == "Ridge": model = Ridge()
                    elif algo == "Lasso": model = Lasso()
                    elif algo == "Random Forest": model = RandomForestRegressor(n_estimators=params.get('n', 100))
                    elif algo == "Gradient Boosting": model = GradientBoostingRegressor(n_estimators=params.get('n', 100))
                    elif algo == "SVR": model = SVR(C=params.get('C', 1.0))
                    elif algo == "Decision Tree": model = DecisionTreeRegressor()
                else:
                    if algo == "Logistic Regression": model = LogisticRegression()
                    elif algo == "Random Forest": model = RandomForestClassifier(n_estimators=params.get('n', 100))
                    elif algo == "Gradient Boosting": model = GradientBoostingClassifier(n_estimators=params.get('n', 100))
                    elif algo == "SVC": model = SVC(C=params.get('C', 1.0))
                    elif algo == "KNN": model = KNeighborsClassifier(n_neighbors=params.get('k', 5))
                    elif algo == "Decision Tree": model = DecisionTreeClassifier()
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Display Results in a Clean Card
                st.markdown('<div class="nav-card">', unsafe_allow_html=True)
                
                if task == "Classification":
                    acc = accuracy_score(y_test, preds)
                    st.metric("Accuracy", f"{acc:.2%}")
                    
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, preds)
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    mse = mean_squared_error(y_test, preds)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("R² Score", f"{r2:.3f}")
                    m2.metric("MAE", f"{mae:.3f}")
                    m3.metric("MSE", f"{mse:.3f}")
                    
                    st.write("**Prediction vs Actual**")
                    fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, template="plotly_dark")
                    fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

# --- 9. PAGE ROUTING ---

if st.session_state.page == 'Home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("DataFlow X")
    st.markdown("### Choose your workflow")
    
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Auto-Pilot")
        st.markdown("One-click cleaning and intelligent model selection.")
        st.markdown('<div class="auto-btn">', unsafe_allow_html=True)
        if st.button("Start Auto-Pilot"): st.session_state.page = "Auto-Pilot"; st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Manual Studio")
        st.markdown("Full control over cleaning, EDA, and model training.")
        if st.button("Enter Studio"): st.session_state.page = "Data Studio"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Auto-Pilot': render_autopilot()
elif st.session_state.page == 'Data Studio': render_data_studio()
elif st.session_state.page == 'EDA Lab': render_eda()
elif st.session_state.page == 'Model Forge': render_model()
