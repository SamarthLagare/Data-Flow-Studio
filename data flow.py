import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from io import BytesIO

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DataFlow Supreme",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State Management
if 'page' not in st.session_state: st.session_state.page = 'Home'
if 'df' not in st.session_state: st.session_state.df = None
if 'history' not in st.session_state: st.session_state.history = []
if 'best_model' not in st.session_state: st.session_state.best_model = None

# --- 2. PREMIUM CSS ---
st.markdown("""
    <style>
    /* IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* CORE THEME */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0e11;
        color: #e2e8f0;
    }
    
    /* LAYOUT */
    .block-container { 
        max-width: 1250px; 
        padding-top: 2.5rem; 
        padding-bottom: 5rem; 
    }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }

    /* HEADINGS */
    h1, h2, h3 { 
        font-weight: 700; 
        color: #f8fafc; 
        letter-spacing: -0.5px;
    }
    h1 { font-size: 3rem; background: -webkit-linear-gradient(45deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    p { color: #94a3b8; font-size: 1.05rem; line-height: 1.6; }

    /* BUTTONS */
    div.stButton > button {
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 10px;
        height: 3.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: #334155;
        border-color: #6366f1;
        color: #6366f1;
    }
    
    /* PRIMARY ACTION BUTTON */
    .primary-btn > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.5);
    }
    .primary-btn > button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px -10px rgba(99, 102, 241, 0.6);
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* METRICS */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #818cf8;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #64748b;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b;
        padding: 4px;
        border-radius: 12px;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 8px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1;
        color: white;
    }

    /* HIDE JUNK */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

def calculate_quality(df):
    if df is None: return 0
    score = 100
    # Penalize for missing
    missing_pct = df.isna().mean().mean()
    score -= (missing_pct * 100)
    # Penalize for duplicates
    dupe_pct = df.duplicated().mean()
    score -= (dupe_pct * 100)
    return max(0, int(score))

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1169/1169345.png", width=50)
    st.title("DataFlow")
    st.caption("v21.0 | Enterprise Edition")
    st.markdown("---")
    
    # NAVIGATION
    nav_options = ["Home", "1. Auto-Pilot", "2. Data Studio", "3. EDA Lab", "4. Model Forge"]
    
    # Sync
    idx_map = {"Home":0, "Auto-Pilot":1, "Data Studio":2, "EDA Lab":3, "Model Forge":4}
    current_idx = idx_map.get(st.session_state.page, 0)
    
    selected = st.radio("MAIN MENU", nav_options, index=current_idx, label_visibility="collapsed")
    
    # Routing
    target_page = selected.replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "")
    if target_page != st.session_state.page:
        st.session_state.page = target_page
        st.rerun()

    # DATA STATUS WIDGET
    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    
    if st.session_state.df is not None:
        quality = calculate_quality(st.session_state.df)
        rows, cols = st.session_state.df.shape
        
        c1, c2 = st.columns(2)
        c1.metric("Rows", rows)
        c2.metric("Cols", cols)
        
        # Quality Bar
        st.write("Data Health")
        st.progress(quality / 100)
        st.caption(f"Quality Score: {quality}%")
        
        if quality < 100:
            st.warning("Optimization Recommended")
    else:
        st.info("Waiting for data...")

# --- 5. COMPONENT: AUTO-PILOT ---
def render_autopilot():
    st.markdown("## ⚡ Auto-Pilot")
    st.markdown("Automated cleaning, analysis, and model tournament.")
    
    if st.session_state.df is None:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])
            if uploaded_file:
                st.session_state.df = load_data(uploaded_file)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        return

    df = st.session_state.df
    
    # CONFIG
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        target = st.selectbox("Target Variable", df.columns)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        start_btn = st.button("🚀 LAUNCH TOURNAMENT")
        st.markdown('</div></div>', unsafe_allow_html=True)

    if start_btn:
        with st.status("🧠 Processing Data Stream...", expanded=True) as status:
            # 1. CLEANING
            status.write("Cleaning pipeline initiated...")
            clean_df = df.copy().drop_duplicates()
            num = clean_df.select_dtypes(include=np.number).columns
            cat = clean_df.select_dtypes(include='object').columns
            clean_df[num] = clean_df[num].fillna(clean_df[num].median())
            clean_df[cat] = clean_df[cat].fillna("Unknown")
            
            # 2. ENCODING
            status.write("Encoding features...")
            ml_df = clean_df.copy()
            le = LabelEncoder()
            for c in ml_df.select_dtypes(include='object').columns:
                ml_df[c] = le.fit_transform(ml_df[c].astype(str))
                
            # 3. SPLIT
            X = ml_df.drop(columns=[target])
            y = ml_df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # 4. TOURNAMENT
            is_class = len(ml_df[target].unique()) < 20
            models = {}
            
            if is_class:
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "SVC": SVC(),
                    "Decision Tree": DecisionTreeClassifier()
                }
                metric = "Accuracy"
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso()
                }
                metric = "R² Score"
            
            results = []
            best_score = -np.inf
            
            for name, model in models.items():
                status.write(f"Training {name}...")
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if is_class: score = accuracy_score(y_test, preds)
                else: score = r2_score(y_test, preds)
                
                results.append({"Model": name, "Score": score})
                if score > best_score:
                    best_score = score
                    st.session_state.best_model = model
            
            status.update(label="Tournament Complete!", state="complete")
            st.session_state.df = clean_df # Save cleaned data
            
            # RESULTS
            res_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
            
            st.divider()
            c_res1, c_res2 = st.columns([1, 1], gap="large")
            
            with c_res1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🏆 Champion Model")
                winner = res_df.iloc[0]
                st.success(f"**{winner['Model']}**")
                st.metric(metric, f"{winner['Score']:.2%}" if is_class else f"{winner['Score']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_res2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Leaderboard")
                fig = px.bar(res_df, x="Score", y="Model", orientation='h', template="plotly_dark", color="Score", title="Performance Comparison")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# --- 6. COMPONENT: DATA STUDIO ---
def render_data_studio():
    st.header("💿 Data Studio")
    
    if st.session_state.df is None:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
            if uploaded_file:
                st.session_state.df = load_data(uploaded_file)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        return

    df = st.session_state.df
    
    # METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", df.shape[0])
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing", df.isna().sum().sum())
    m4.metric("Duplicates", df.duplicated().sum())
    
    st.markdown("### Workbench")
    
    t1, t2, t3 = st.tabs(["Cleaning", "Transformation", "Export"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Smart Cleaning**")
            col = st.selectbox("Target Column", df.columns)
            if st.button("Fill Missing (Median/Mode)"):
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.session_state.df[col] = df[col].fillna(df[col].median())
                else:
                    st.session_state.df[col] = df[col].fillna(df[col].mode()[0])
                st.rerun()
        with c2:
            st.markdown("**Deduplication**")
            if st.button("Remove All Duplicates"):
                st.session_state.df = df.drop_duplicates()
                st.rerun()

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Filter Data**")
            fil_col = st.selectbox("Filter Column", df.columns)
            val = st.text_input("Value to Keep")
            if st.button("Apply Filter"):
                # Try numeric conversion
                try: val = float(val)
                except: pass
                st.session_state.df = df[df[fil_col] == val]
                st.rerun()
        with c2:
            st.markdown("**Drop Column**")
            drop_c = st.selectbox("Column to Drop", df.columns)
            if st.button("Delete Column"):
                st.session_state.df = df.drop(columns=[drop_c])
                st.rerun()

    with t3:
        st.download_button("⬇️ Download Processed Dataset", convert_df(df), "data_flow_export.csv", "text/csv")
        
    st.markdown("### Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

# --- 7. COMPONENT: EDA LAB ---
def render_eda():
    st.header("📊 EDA Lab")
    if st.session_state.df is None: st.warning("Upload data first"); return
    
    df = st.session_state.df
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### Configuration")
        chart = st.selectbox("Chart Type", ["Histogram", "Box", "Violin", "Scatter", "Line", "Bar", "Area", "Pie", "Heatmap", "3D Scatter", "Funnel"])
    with c2:
        st.markdown("### Axes")
        ac1, ac2, ac3 = st.columns(3)
        x = ac1.selectbox("X Axis", all_cols)
        y = ac2.selectbox("Y Axis", num_cols if chart != "Pie" else all_cols)
        c = ac3.selectbox("Color", [None] + all_cols)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PLOTTING ENGINE
    try:
        if chart == "Histogram": fig = px.histogram(df, x=x, color=c, template="plotly_dark", marginal="box")
        elif chart == "Box": fig = px.box(df, x=x, y=y, color=c, template="plotly_dark")
        elif chart == "Violin": fig = px.violin(df, x=x, y=y, color=c, box=True, template="plotly_dark")
        elif chart == "Scatter": fig = px.scatter(df, x=x, y=y, color=c, template="plotly_dark", size=y if y in num_cols else None)
        elif chart == "Line": fig = px.line(df, x=x, y=y, color=c, template="plotly_dark")
        elif chart == "Bar": fig = px.bar(df, x=x, y=y, color=c, template="plotly_dark")
        elif chart == "Area": fig = px.area(df, x=x, y=y, color=c, template="plotly_dark")
        elif chart == "Pie": fig = px.pie(df, names=x, template="plotly_dark")
        elif chart == "Heatmap": fig = px.density_heatmap(df, x=x, y=y, template="plotly_dark")
        elif chart == "3D Scatter": 
            z = st.selectbox("Z Axis", num_cols)
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=c, template="plotly_dark")
        elif chart == "Funnel": fig = px.funnel(df, x=x, y=y, color=c, template="plotly_dark")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Auto-Stats
        with st.expander("Show Statistics for Selected Data"):
            st.write(df[[x, y] if y else [x]].describe())
            
    except Exception as e:
        st.error(f"Could not render chart: {e}")

# --- 8. COMPONENT: MODEL FORGE ---
def render_model():
    st.header("🧠 Model Forge")
    if st.session_state.df is None: st.warning("Upload data first"); return
    
    df = st.session_state.df.copy().dropna()
    
    # Auto-Encode
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c].astype(str))
        
    c1, c2 = st.columns([1, 2], gap="large")
    
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 1. Setup")
        target = st.selectbox("Target", df.columns)
        feats = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        
        # Type Detect
        task = "Regression" if len(df[target].unique()) > 20 else "Classification"
        st.caption(f"Detected Task: **{task}**")
        
        st.markdown("### 2. Model")
        if task == "Regression":
            algo = st.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Gradient Boosting", "AdaBoost", "SVR", "Ridge", "Lasso", "Decision Tree", "KNN"])
        else:
            algo = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "Gradient Boosting", "AdaBoost", "SVC", "Decision Tree", "KNN", "Gaussian NB"])
            
        st.markdown("### 3. Tuning")
        split = st.slider("Train Split", 0.1, 0.9, 0.8)
        
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        train = st.button("🚀 Train Model")
        st.markdown('</div></div>', unsafe_allow_html=True)

    with c2:
        if train:
            with st.spinner("Training..."):
                X = df[feats]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model = None
                # MODEL FACTORY
                if algo == "Linear Regression": model = LinearRegression()
                elif algo == "Logistic Regression": model = LogisticRegression()
                elif algo == "Random Forest": 
                    model = RandomForestRegressor() if task == "Regression" else RandomForestClassifier()
                elif algo == "Gradient Boosting":
                    model = GradientBoostingRegressor() if task == "Regression" else GradientBoostingClassifier()
                elif algo == "AdaBoost":
                    model = AdaBoostRegressor() if task == "Regression" else AdaBoostClassifier()
                elif algo == "SVR": model = SVR()
                elif algo == "SVC": model = SVC()
                elif algo == "KNN":
                    model = KNeighborsRegressor() if task == "Regression" else KNeighborsClassifier()
                elif algo == "Ridge": model = Ridge()
                elif algo == "Lasso": model = Lasso()
                elif algo == "Decision Tree":
                    model = DecisionTreeRegressor() if task == "Regression" else DecisionTreeClassifier()
                elif algo == "Gaussian NB": model = GaussianNB()
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # METRICS
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"### Performance: {algo}")
                
                if task == "Classification":
                    acc = accuracy_score(y_test, preds)
                    st.metric("Accuracy", f"{acc:.2%}")
                    fig = px.imshow(confusion_matrix(y_test, preds), text_auto=True, title="Confusion Matrix", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    c_m1, c_m2 = st.columns(2)
                    c_m1.metric("R² Score", f"{r2:.4f}")
                    c_m2.metric("MAE", f"{mae:.4f}")
                    
                    fig = px.scatter(x=y_test, y=preds, labels={'x':'Actual', 'y':'Predicted'}, title="Actual vs Predicted", template="plotly_dark")
                    fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # DOWNLOAD
                b_model = BytesIO()
                pickle.dump(model, b_model)
                st.download_button("⬇️ Download Trained Model", b_model.getvalue(), "model.pkl")

# --- 9. PAGE ROUTING ---

if st.session_state.page == 'Home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>DataFlow Supreme</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 50px;'>The Ultimate AI-Powered Analytics Platform</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💿 Data Studio")
        st.write("Advanced cleaning, filtering, and quality scoring.")
        if st.button("Launch Studio"): st.session_state.page = "Data Studio"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 EDA Lab")
        st.write("12+ Interactive visualizations with instant insights.")
        if st.button("Launch Lab"): st.session_state.page = "EDA Lab"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Model Forge")
        st.write("Train 10+ algorithms manually or use Auto-Pilot.")
        if st.button("Launch Forge"): st.session_state.page = "Model Forge"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Auto-Pilot Banner
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="text-align:center; border-color: #6366f1;">', unsafe_allow_html=True)
    st.markdown("### ⚡ Want speed?")
    st.write("Use Auto-Pilot to clean, analyze, and find the best model in one click.")
    st.markdown('<div class="auto-btn" style="width: 50%; margin: 0 auto;">', unsafe_allow_html=True)
    if st.button("Start Auto-Pilot"): st.session_state.page = "Auto-Pilot"; st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

elif st.session_state.page == 'Auto-Pilot': render_autopilot()
elif st.session_state.page == 'Data Studio': render_data_studio()
elif st.session_state.page == 'EDA Lab': render_eda()
elif st.session_state.page == 'Model Forge': render_model()
