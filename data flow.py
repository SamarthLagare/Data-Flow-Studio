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
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix

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

# --- 2. CLEAN & SIMPLE CSS ---
st.markdown("""
    <style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        background-color: #0E1117;
        color: #e0e0e0;
    }
    
    /* Layout */
    .block-container { 
        max-width: 1100px; 
        padding-top: 2rem; 
        padding-bottom: 5rem; 
    }

    /* Cards */
    .nav-card {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Primary Button */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        height: 3rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Auto Button */
    .auto-btn > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid #333;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #3b82f6; font-size: 1.4rem; }
    div[data-testid="stMetricLabel"] { color: #888; font-size: 0.9rem; }
    
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
    st.title("DataFlow")
    st.markdown("Analytics Suite")
    
    nav_options = ["Home", "Auto-Pilot", "Data Studio", "EDA Lab", "Model Forge"]
    
    # Sync Sidebar
    ix = 0
    if st.session_state.page == "Auto-Pilot": ix = 1
    elif st.session_state.page == "Data Studio": ix = 2
    elif st.session_state.page == "EDA Lab": ix = 3
    elif st.session_state.page == "Model Forge": ix = 4
    
    selected = st.radio("Navigate", nav_options, index=ix, label_visibility="collapsed")
    
    if selected == "Home": st.session_state.page = "Home"
    elif selected == "Auto-Pilot": st.session_state.page = "Auto-Pilot"
    elif selected == "Data Studio": st.session_state.page = "Data Studio"
    elif selected == "EDA Lab": st.session_state.page = "EDA Lab"
    elif selected == "Model Forge": st.session_state.page = "Model Forge"

    st.markdown("---")
    if st.session_state.df is not None:
        r, c = st.session_state.df.shape
        st.success(f"Data: {r} rows, {c} cols")
    else:
        st.info("No Data Loaded")

# --- 5. COMPONENT: AUTO-PILOT ---
def render_autopilot():
    st.header("⚡ Auto-Pilot")
    st.markdown("One-click analysis and modeling.")
    
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.rerun()
        return

    df = st.session_state.df
    
    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.selectbox("Target Variable", df.columns)
        st.markdown('<div class="auto-btn">', unsafe_allow_html=True)
        if st.button("🚀 Start Auto-Pilot"):
            with st.spinner("Analyzing..."):
                # 1. Clean
                clean_df = df.drop_duplicates()
                num_cols = clean_df.select_dtypes(include=np.number).columns
                clean_df[num_cols] = clean_df[num_cols].fillna(clean_df[num_cols].median())
                cat_cols = clean_df.select_dtypes(include='object').columns
                clean_df[cat_cols] = clean_df[cat_cols].fillna("Unknown")
                
                # 2. Encode
                ml_df = clean_df.copy()
                le = LabelEncoder()
                for c in ml_df.select_dtypes(include='object').columns:
                    ml_df[c] = le.fit_transform(ml_df[c].astype(str))
                
                # 3. Smart Model Selection
                X = ml_df.drop(columns=[target])
                y = ml_df[target]
                unique_y = len(ml_df[target].unique())
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Logic: Use Linear/Logistic for small data (faster/interpretable), RF for complex
                is_class = unique_y < 20
                rows = len(ml_df)
                
                if is_class:
                    if rows < 1000:
                        model = LogisticRegression()
                        algo_name = "Logistic Regression (Best for small data)"
                    else:
                        model = RandomForestClassifier(n_estimators=100)
                        algo_name = "Random Forest (Best for complex data)"
                        
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds)
                    metric = "Accuracy"
                else:
                    # Check linearity correlation roughly
                    corr = abs(ml_df.corr()[target].mean())
                    if corr > 0.5:
                        model = LinearRegression()
                        algo_name = "Linear Regression (High correlation detected)"
                    else:
                        model = RandomForestRegressor(n_estimators=100)
                        algo_name = "Random Forest (Non-linear patterns)"
                        
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = r2_score(y_test, preds)
                    metric = "R² Score"

                st.session_state.auto_res = {
                    "score": score, "metric": metric, "algo": algo_name, 
                    "model": model, "clean_df": clean_df
                }
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if 'auto_res' in st.session_state:
            res = st.session_state.auto_res
            st.markdown('<div class="nav-card">', unsafe_allow_html=True)
            st.metric(res["metric"], f"{res['score']:.2%}" if res["metric"] == "Accuracy" else f"{res['score']:.3f}")
            st.caption(f"Selected Algorithm: **{res['algo']}**")
            
            # Simple Feature Importance
            if hasattr(res['model'], 'feature_importances_'):
                imp = pd.DataFrame({'Feature': df.drop(columns=[target]).columns, 
                                  'Importance': res['model'].feature_importances_}).sort_values('Importance', ascending=True).tail(5)
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h', template="plotly_dark", title="Top Drivers")
                st.plotly_chart(fig, use_container_width=True, key="auto_imp")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 6. COMPONENT: DATA STUDIO ---
def render_data():
    st.header("Data Studio")
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.rerun()
    else:
        df = st.session_state.df
        
        t1, t2 = st.tabs(["Cleaning", "Export"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Missing Values**")
                col = st.selectbox("Column", df.columns, key="cl_c")
                act = st.selectbox("Action", ["Fill Median", "Fill Mode", "Drop Rows"], key="cl_a")
                if st.button("Apply"):
                    if act == "Drop Rows": st.session_state.df = df.dropna(subset=[col])
                    elif act == "Fill Median": st.session_state.df[col] = df[col].fillna(df[col].median())
                    elif act == "Fill Mode": st.session_state.df[col] = df[col].fillna(df[col].mode()[0])
                    st.rerun()
            with c2:
                st.markdown("**Duplicates**")
                if st.button("Remove Duplicates"):
                    st.session_state.df = df.drop_duplicates()
                    st.rerun()
                    
        with t2:
            st.download_button("Download CSV", convert_df(df), "clean.csv", "text/csv")
            
        st.markdown("### Preview")
        st.dataframe(df.head(), use_container_width=True)

# --- 7. COMPONENT: EDA LAB ---
def render_eda():
    st.header("EDA Lab")
    if st.session_state.df is None: st.info("No Data"); return
    
    df = st.session_state.df
    cols = df.columns.tolist()
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("**Settings**")
        type_ = st.selectbox("Chart", ["Histogram", "Box", "Scatter", "Line", "Bar", "Violin", "Pie"])
        x = st.selectbox("X Axis", cols)
        y = st.selectbox("Y Axis", cols) if type_ not in ["Histogram", "Pie"] else None
        c = st.selectbox("Color", [None] + cols)
    
    with c2:
        if type_ == "Histogram": fig = px.histogram(df, x=x, color=c, template="plotly_dark")
        elif type_ == "Box": fig = px.box(df, x=x, y=y, color=c, template="plotly_dark")
        elif type_ == "Scatter": fig = px.scatter(df, x=x, y=y, color=c, template="plotly_dark")
        elif type_ == "Line": fig = px.line(df, x=x, y=y, color=c, template="plotly_dark")
        elif type_ == "Bar": fig = px.bar(df, x=x, y=y, color=c, template="plotly_dark")
        elif type_ == "Violin": fig = px.violin(df, x=x, y=y, color=c, template="plotly_dark")
        elif type_ == "Pie": fig = px.pie(df, names=x, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, key="eda_chart")

# --- 8. COMPONENT: MODEL FORGE (IMPROVED) ---
def render_model():
    st.header("Model Forge")
    if st.session_state.df is None: st.info("No Data"); return
    
    df = st.session_state.df.copy().dropna()
    
    # Encode
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c].astype(str))
        
    # --- 1. DATA STATISTICS PANEL ---
    st.markdown('<div class="nav-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", df.shape[0])
    m2.metric("Features", df.shape[1])
    m3.metric("Numeric Cols", len(df.select_dtypes(include=np.number).columns))
    m4.metric("Categorical Cols", len(st.session_state.df.select_dtypes(include='object').columns))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. CONFIGURATION ---
    c_conf, c_res = st.columns([1, 2], gap="large")
    
    with c_conf:
        st.subheader("Setup")
        target = st.selectbox("Target", df.columns)
        feats = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        
        # Simple Task Detection
        task = "Regression"
        if len(df[target].unique()) < 20: task = "Classification"
        st.info(f"Task: {task}")
        
        algo = st.selectbox("Algorithm", 
                            ["Random Forest", "Linear/Logistic Regression", "Decision Tree"] if task == "Classification" 
                            else ["Random Forest", "Linear Regression", "Decision Tree"])
        
        split = st.slider("Split %", 0.1, 0.5, 0.2)
        
        if st.button("Train Model"):
            with st.spinner("Training..."):
                X = df[feats]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
                
                model = None
                if task == "Classification":
                    if "Random" in algo: model = RandomForestClassifier()
                    elif "Logistic" in algo: model = LogisticRegression()
                    else: model = DecisionTreeClassifier()
                    model.fit(X_train, y_train)
                    metric = accuracy_score(y_test, model.predict(X_test))
                    lbl = "Accuracy"
                else:
                    if "Random" in algo: model = RandomForestRegressor()
                    elif "Linear" in algo: model = LinearRegression()
                    else: model = DecisionTreeRegressor()
                    model.fit(X_train, y_train)
                    metric = r2_score(y_test, model.predict(X_test))
                    lbl = "R² Score"
                
                st.session_state.model_res = {"model": model, "score": metric, "lbl": lbl, "y_test": y_test, "preds": model.predict(X_test), "task": task}

    with c_res:
        st.subheader("Results")
        if 'model_res' in st.session_state:
            res = st.session_state.model_res
            st.markdown('<div class="nav-card">', unsafe_allow_html=True)
            st.metric(res["lbl"], f"{res['score']:.3f}")
            
            # Simple Visuals
            if res["task"] == "Regression":
                fig = px.scatter(x=res["y_test"], y=res["preds"], labels={'x': 'Actual', 'y': 'Predicted'}, template="plotly_dark", title="Accuracy Plot")
                fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=min(res["y_test"]), y0=min(res["y_test"]), x1=max(res["y_test"]), y1=max(res["y_test"]))
                st.plotly_chart(fig, use_container_width=True, key="forge_reg")
            else:
                cm = confusion_matrix(res["y_test"], res["preds"])
                fig = px.imshow(cm, text_auto=True, template="plotly_dark", title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True, key="forge_cm")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 9. ROUTING ---
if st.session_state.page == 'Home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("DataFlow")
    st.markdown("### Analytics Simplified")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 💿 Data")
        if st.button("Studio"): st.session_state.page = "Data Studio"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Visuals")
        if st.button("Lab"): st.session_state.page = "EDA Lab"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Models")
        if st.button("Forge"): st.session_state.page = "Model Forge"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Auto-Pilot': render_autopilot()
elif st.session_state.page == 'Data Studio': render_data()
elif st.session_state.page == 'EDA Lab': render_eda()
elif st.session_state.page == 'Model Forge': render_model()
