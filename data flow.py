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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DataFlow Aether",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State Management
if 'page' not in st.session_state: st.session_state.page = 'Home'
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None

# --- 2. GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* BACKGROUND GRADIENT */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(30, 41, 59) 90%);
        font-family: 'Inter', sans-serif;
        color: white;
    }
    
    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 24px;
        border-radius: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* SIDEBAR GLASS */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* HEADINGS */
    h1 {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    h2, h3 { color: #f1f5f9; font-weight: 600; }
    p { color: #94a3b8; }

    /* BUTTONS */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 12px;
        height: 3.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: #00c6ff;
        box-shadow: 0 0 15px rgba(0, 198, 255, 0.3);
    }
    
    /* PRIMARY GLOW BUTTON */
    .glow-btn > button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(0, 114, 255, 0.4);
    }
    .glow-btn > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 114, 255, 0.6);
    }

    /* INPUTS & SELECTS */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 5px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #0072ff;
        color: white;
        border-radius: 8px;
    }

    /* REMOVE JUNK */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* METRICS */
    div[data-testid="stMetricValue"] { color: #00c6ff; }
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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🔮 DataFlow")
    st.caption("Aether Edition")
    st.markdown("---")
    
    pages = ["Home", "Auto-Pilot", "Data Studio", "EDA Lab", "Model Forge"]
    
    # Sync Logic
    curr = st.session_state.page
    idx = pages.index(curr) if curr in pages else 0
    sel = st.radio("MENU", pages, index=idx, label_visibility="collapsed")
    
    if sel != st.session_state.page:
        st.session_state.page = sel
        st.rerun()

    st.markdown("---")
    if st.session_state.df is not None:
        r, c = st.session_state.df.shape
        st.success(f"Data: {r} rows | {c} cols")
    else:
        st.info("No Data Loaded")

# --- 5. COMPONENT: OMNI-PILOT (AUTOMATE EVERYTHING) ---
def render_autopilot():
    st.header("⚡ Omni-Pilot")
    st.markdown("One click to Clean, Visualize, and Model.")
    
    if st.session_state.df is None:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            f = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
            if f:
                st.session_state.df = load_data(f)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        return

    df = st.session_state.df
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Config")
        target = st.selectbox("Target Variable", df.columns)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
        if st.button("🚀 LAUNCH OMNI-PILOT"):
            st.session_state.run_auto = True
        st.markdown('</div></div>', unsafe_allow_html=True)

    if st.session_state.get('run_auto'):
        with st.status("🔮 AI Processing Pipeline...", expanded=True) as status:
            
            # --- 1. AUTO CLEAN ---
            status.write("🧹 Scrubbing Data...")
            clean_df = df.copy().drop_duplicates()
            num = clean_df.select_dtypes(include=np.number).columns
            cat = clean_df.select_dtypes(include='object').columns
            clean_df[num] = clean_df[num].fillna(clean_df[num].median())
            clean_df[cat] = clean_df[cat].fillna("Unknown")
            st.session_state.df = clean_df
            
            # --- 2. AUTO VIZ ---
            status.write("📊 Generating 10 Analytics Views...")
            figs = []
            
            # 1. Distribution
            figs.append(px.histogram(clean_df, x=target, title=f"1. Target Distribution ({target})", template="plotly_dark", color_discrete_sequence=['#00c6ff']))
            
            # 2. Correlation
            if len(num) > 1:
                figs.append(px.imshow(clean_df[num].corr(), text_auto=True, title="2. Correlation Heatmap", template="plotly_dark", color_continuous_scale='Viridis'))
            
            # 3. Scatter (First 2 num cols)
            if len(num) >= 2:
                figs.append(px.scatter(clean_df, x=num[0], y=num[1], color=target if target in cat else None, title=f"3. Scatter: {num[0]} vs {num[1]}", template="plotly_dark"))
            
            # 4. Box Plot (Target vs Cat)
            if len(cat) > 0 and target in num:
                figs.append(px.box(clean_df, x=cat[0], y=target, title=f"4. Box Plot: {target} by {cat[0]}", template="plotly_dark"))
            
            # 5. Line (Index vs Target)
            figs.append(px.line(clean_df, y=target, title="5. Target Trend (Index)", template="plotly_dark"))
            
            # 6. Violin
            figs.append(px.violin(clean_df, y=target, box=True, title="6. Target Density (Violin)", template="plotly_dark"))
            
            # 7. 3D Scatter
            if len(num) >= 3:
                figs.append(px.scatter_3d(clean_df, x=num[0], y=num[1], z=num[2], color=target if target in cat else None, title="7. 3D Analysis", template="plotly_dark"))
            
            # 8. Density Heatmap
            if len(num) >= 2:
                figs.append(px.density_heatmap(clean_df, x=num[0], y=num[1], title="8. Density Heatmap", template="plotly_dark"))
                
            # 9. Bar Count (if cat target) or Bar Avg
            if target in cat:
                figs.append(px.bar(clean_df[target].value_counts(), title="9. Class Balance", template="plotly_dark"))
            
            # 10. Pie
            if target in cat:
                figs.append(px.pie(clean_df, names=target, title="10. Target Composition", template="plotly_dark"))

            # --- 3. AUTO MODEL ---
            status.write("🧠 Running Model Tournament...")
            ml_df = clean_df.copy()
            le = LabelEncoder()
            for c in ml_df.select_dtypes(include='object').columns:
                ml_df[c] = le.fit_transform(ml_df[c].astype(str))
                
            X = ml_df.drop(columns=[target])
            y = ml_df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            is_class = len(ml_df[target].unique()) < 15
            
            models = {
                "Random Forest": RandomForestClassifier() if is_class else RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingClassifier() if is_class else GradientBoostingRegressor(),
                "Linear/Logistic": LogisticRegression() if is_class else LinearRegression(),
                "Decision Tree": DecisionTreeClassifier() if is_class else DecisionTreeRegressor(),
                "K-Neighbors": KNeighborsClassifier() if is_class else KNeighborsRegressor()
            }
            
            best_score = -999
            best_model = None
            leaderboard = []
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds) if is_class else r2_score(y_test, preds)
                leaderboard.append({"Model": name, "Score": score})
                if score > best_score:
                    best_score = score
                    best_model = model
            
            st.session_state.model = best_model
            status.update(label="Omni-Pilot Complete!", state="complete")

        # --- REPORT DISPLAY ---
        st.markdown("### 🏆 Results")
        
        # Leaderboard
        c_lead, c_win = st.columns([2, 1])
        with c_lead:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            lb_df = pd.DataFrame(leaderboard).sort_values("Score", ascending=False)
            fig_lb = px.bar(lb_df, x="Score", y="Model", orientation='h', color="Score", template="plotly_dark", title="Model Tournament Results")
            st.plotly_chart(fig_lb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c_win:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric("🏆 Winner", lb_df.iloc[0]['Model'])
            st.metric("Score", f"{lb_df.iloc[0]['Score']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # 10 Graphs Gallery
        st.markdown("### 📸 Auto-Generated Insights")
        for i in range(0, len(figs), 2):
            gc1, gc2 = st.columns(2)
            with gc1:
                if i < len(figs): st.plotly_chart(figs[i], use_container_width=True)
            with gc2:
                if i+1 < len(figs): st.plotly_chart(figs[i+1], use_container_width=True)

# --- 6. COMPONENT: DATA STUDIO ---
def render_studio():
    st.header("💿 Data Studio")
    if st.session_state.df is None:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            f = st.file_uploader("Upload", type=['csv', 'xlsx'])
            if f:
                st.session_state.df = load_data(f)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        return

    df = st.session_state.df
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", df.shape[0])
    m2.metric("Missing", df.isna().sum().sum())
    m3.metric("Duplicates", df.duplicated().sum())
    st.markdown('</div>', unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["Cleaning", "Export"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Imputation**")
            col = st.selectbox("Column", df.columns)
            if st.button("Fill Median"):
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.session_state.df[col] = df[col].fillna(df[col].median())
                    st.rerun()
        with c2:
            st.markdown("**Maintenance**")
            if st.button("Drop Duplicates"):
                st.session_state.df = df.drop_duplicates()
                st.rerun()
                
    with t2:
        st.download_button("Download CSV", convert_df(df), "clean_data.csv")
    
    st.dataframe(df.head(50), use_container_width=True)

# --- 7. COMPONENT: EDA LAB ---
def render_eda():
    st.header("📊 EDA Lab")
    if st.session_state.df is None: st.warning("No Data"); return
    
    df = st.session_state.df
    cols = df.columns.tolist()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        chart = st.selectbox("Chart Type", ["Histogram", "Box", "Scatter", "Line", "Bar", "Pie", "Heatmap", "3D Scatter"])
    with c2:
        ac1, ac2, ac3 = st.columns(3)
        x = ac1.selectbox("X", cols)
        y = ac2.selectbox("Y", cols)
        c = ac3.selectbox("Color", [None] + cols)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if chart == "Histogram": fig = px.histogram(df, x=x, color=c, template="plotly_dark")
    elif chart == "Box": fig = px.box(df, x=x, y=y, color=c, template="plotly_dark")
    elif chart == "Scatter": fig = px.scatter(df, x=x, y=y, color=c, template="plotly_dark")
    elif chart == "Line": fig = px.line(df, x=x, y=y, color=c, template="plotly_dark")
    elif chart == "Bar": fig = px.bar(df, x=x, y=y, color=c, template="plotly_dark")
    elif chart == "Pie": fig = px.pie(df, names=x, template="plotly_dark")
    elif chart == "Heatmap": fig = px.density_heatmap(df, x=x, y=y, template="plotly_dark")
    elif chart == "3D Scatter": fig = px.scatter_3d(df, x=x, y=y, z=c if c else x, color=c, template="plotly_dark")
    
    st.plotly_chart(fig, use_container_width=True)

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
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        target = st.selectbox("Target", df.columns)
        feats = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        algo = st.selectbox("Algorithm", ["Random Forest", "Linear/Logistic", "Gradient Boosting"])
        st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
        train = st.button("Train Manually")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
    if train:
        X, y = df[feats], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        is_class = len(df[target].unique()) < 15
        if algo == "Random Forest": model = RandomForestClassifier() if is_class else RandomForestRegressor()
        elif algo == "Linear/Logistic": model = LogisticRegression() if is_class else LinearRegression()
        else: model = GradientBoostingClassifier() if is_class else GradientBoostingRegressor()
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric("Model Score", f"{score:.4f}")
            st.success("Training Successful")
            
            b = BytesIO()
            pickle.dump(model, b)
            st.download_button("Download Model", b.getvalue(), "model.pkl")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 9. ROUTING ---
if st.session_state.page == 'Home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("DataFlow Aether")
    st.markdown("### The Ultimate AI Analytics Platform")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Omni-Pilot")
        st.caption("Clean, Visualize & Model in 1 Click.")
        st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
        if st.button("Start Auto-Pilot"): st.session_state.page = "Auto-Pilot"; st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Manual Studio")
        st.caption("Granular control over every step.")
        if st.button("Enter Studio"): st.session_state.page = "Data Studio"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Auto-Pilot': render_autopilot()
elif st.session_state.page == 'Data Studio': render_studio()
elif st.session_state.page == 'EDA Lab': render_eda()
elif st.session_state.page == 'Model Forge': render_model()
