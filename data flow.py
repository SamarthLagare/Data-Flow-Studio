import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import pickle
from io import BytesIO

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, mean_absolute_error, 
                             confusion_matrix, classification_report)

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Data Nexus Ultimate",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State Management
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    /* DARK THEME CORE */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
        color: #e0e0e0;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* CONTAINERS */
    .block-container { max-width: 95%; padding-top: 2rem; }
    
    .data-card {
        background-color: #1f242d;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 15px;
    }

    /* BUTTONS */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        height: 2.8em;
        font-weight: 600;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
        transform: translateY(-2px);
    }
    
    /* SECONDARY BUTTONS (Gray) */
    .secondary-btn > button {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
    }

    /* METRICS */
    div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.6rem; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }

    /* INPUTS */
    div[data-baseweb="select"] > div { background-color: #0d1117; border-color: #30363d; }
    
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

def download_obj(obj, name):
    output = BytesIO()
    pickle.dump(obj, output)
    return output.getvalue()

def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 4. DATA STUDIO (CLEANING & PREP) ---
def render_studio():
    st.markdown("## 💾 Data Studio")
    
    c1, c2 = st.columns([1, 3], gap="large")
    
    with c1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("### Import")
        file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'], label_visibility="collapsed")
        if file:
            st.session_state.df = load_data(file)
            st.success(f"Loaded: {file.name}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.df is not None:
            df = st.session_state.df
            
            # --- CLEANING TOOLS ---
            with st.expander("🧹 Data Cleaning", expanded=True):
                st.caption("Missing Values")
                impute_method = st.selectbox("Imputation Method", ["Drop Rows", "Fill Mean", "Fill Median", "Fill Mode", "Fill Zero"])
                
                if st.button("Apply Imputation"):
                    if impute_method == "Drop Rows":
                        st.session_state.df = df.dropna()
                    elif impute_method == "Fill Zero":
                        st.session_state.df = df.fillna(0)
                    else:
                        num_cols = df.select_dtypes(include=np.number).columns
                        if impute_method == "Fill Mean":
                            st.session_state.df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                        elif impute_method == "Fill Median":
                            st.session_state.df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                        elif impute_method == "Fill Mode":
                            st.session_state.df = df.fillna(df.mode().iloc[0])
                    st.toast("Imputation Applied!", icon="✅")
                    st.rerun()

                st.divider()
                st.caption("Duplicates")
                if st.button("Remove Duplicates"):
                    orig = len(df)
                    st.session_state.df = df.drop_duplicates()
                    st.toast(f"Removed {orig - len(st.session_state.df)} duplicates", icon="🗑️")
                    st.rerun()

            # --- COLUMN MANAGER ---
            with st.expander("🏗️ Column Manager"):
                cols = st.multiselect("Select Columns to Drop", df.columns)
                if st.button("Drop Columns"):
                    st.session_state.df = df.drop(columns=cols)
                    st.rerun()
                
                st.divider()
                st.caption("Type Conversion")
                c_name = st.selectbox("Col", df.columns, key="conv_col")
                to_type = st.selectbox("To", ["Numeric", "String", "Datetime", "Category"], key="conv_type")
                if st.button("Convert Type"):
                    try:
                        if to_type == "Numeric": st.session_state.df[c_name] = pd.to_numeric(df[c_name], errors='coerce')
                        elif to_type == "String": st.session_state.df[c_name] = df[c_name].astype(str)
                        elif to_type == "Datetime": st.session_state.df[c_name] = pd.to_datetime(df[c_name], errors='coerce')
                        elif to_type == "Category": st.session_state.df[c_name] = df[c_name].astype('category')
                        st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

    with c2:
        if st.session_state.df is not None:
            # INFO BAR
            df = st.session_state.df
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", df.shape[0])
            m2.metric("Columns", df.shape[1])
            m3.metric("Missing", df.isna().sum().sum())
            m4.metric("Duplicates", df.duplicated().sum())
            
            # TABS FOR VIEWING
            t1, t2, t3 = st.tabs(["DataFrame", "Statistics", "Column Types"])
            with t1: st.dataframe(df, use_container_width=True, height=500)
            with t2: st.dataframe(df.describe(), use_container_width=True)
            with t3: st.dataframe(df.dtypes.astype(str), use_container_width=True)
            
            # EXPORT
            st.download_button("⬇️ Download Current CSV", download_csv(df), "processed_data.csv", "text/csv")
        else:
            st.info("Please upload a dataset to begin.")

# --- 5. EDA LAB (VISUALIZATION) ---
def render_eda():
    st.markdown("## 📊 EDA Lab")
    
    if st.session_state.df is None:
        st.warning("Upload data in Data Studio first.")
        return

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    t1, t2, t3, t4 = st.tabs(["Univariate", "Bivariate", "Multivariate", "Correlation"])

    # 1. UNIVARIATE
    with t1:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            u_feat = st.selectbox("Feature", all_cols)
            u_type = st.radio("Plot Type", ["Histogram", "Box Plot", "Violin Plot", "Bar Chart"])
            u_color = st.selectbox("Color Group", [None] + cat_cols, key="u_col")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            if u_type == "Histogram": fig = px.histogram(df, x=u_feat, color=u_color, marginal="box", template="plotly_dark")
            elif u_type == "Box Plot": fig = px.box(df, y=u_feat, color=u_color, template="plotly_dark")
            elif u_type == "Violin Plot": fig = px.violin(df, y=u_feat, color=u_color, box=True, template="plotly_dark")
            elif u_type == "Bar Chart": fig = px.bar(df[u_feat].value_counts().reset_index(), x='index', y=u_feat, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # 2. BIVARIATE
    with t2:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            bi_x = st.selectbox("X Axis", all_cols, key="bi_x")
            bi_y = st.selectbox("Y Axis", num_cols, key="bi_y")
            bi_type = st.radio("Style", ["Scatter", "Line", "Bar"], horizontal=True)
            bi_color = st.selectbox("Color Overlay", [None] + all_cols, key="bi_c")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            if bi_type == "Scatter": fig = px.scatter(df, x=bi_x, y=bi_y, color=bi_color, template="plotly_dark")
            elif bi_type == "Line": fig = px.line(df, x=bi_x, y=bi_y, color=bi_color, template="plotly_dark")
            elif bi_type == "Bar": fig = px.bar(df, x=bi_x, y=bi_y, color=bi_color, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # 3. MULTIVARIATE
    with t3:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            mul_x = st.selectbox("X", num_cols, key="m_x")
            mul_y = st.selectbox("Y", num_cols, key="m_y")
            mul_z = st.selectbox("Z (3D Only)", num_cols, key="m_z")
            mul_c = st.selectbox("Color", [None] + all_cols, key="m_c")
            mul_s = st.selectbox("Size", [None] + num_cols, key="m_s")
            plot_3d = st.toggle("Enable 3D Plot")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            if plot_3d:
                fig = px.scatter_3d(df, x=mul_x, y=mul_y, z=mul_z, color=mul_c, size=mul_s, template="plotly_dark")
            else:
                fig = px.scatter(df, x=mul_x, y=mul_y, color=mul_c, size=mul_s, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # 4. CORRELATION
    with t4:
        if num_cols:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark", aspect="auto")
            st.plotly_chart(fig, use_container_width=True, height=700)
        else:
            st.warning("No numeric columns for correlation.")

# --- 6. MODEL FORGE (MACHINE LEARNING) ---
def render_forge():
    st.markdown("## 🧠 Model Forge")
    
    if st.session_state.df is None:
        st.warning("Upload data first.")
        return

    df = st.session_state.df.copy().dropna()
    
    # Encoder
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    c_sets, c_res = st.columns([1, 3], gap="large")

    with c_sets:
        st.markdown("### ⚙️ Configuration")
        
        target = st.selectbox("Target (Label)", df.columns)
        feats = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        
        task = st.radio("Problem Type", ["Regression", "Classification"], horizontal=True)
        
        st.divider()
        st.markdown("### 🤖 Algorithm")
        
        model_params = {}
        
        if task == "Regression":
            algo = st.selectbox("Model", ["Linear Regression", "Random Forest", "SVR (SVM)", "Neural Network (MLP)", "Gradient Boosting"])
            
            if algo == "Random Forest":
                model_params['n_estimators'] = st.slider("Trees", 10, 500, 100)
                model_params['max_depth'] = st.slider("Max Depth", 1, 50, 10)
            elif algo == "SVR (SVM)":
                model_params['C'] = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
                model_params['kernel'] = st.selectbox("Kernel", ['linear', 'poly', 'rbf'])
            elif algo == "Gradient Boosting":
                model_params['lr'] = st.slider("Learning Rate", 0.01, 0.5, 0.1)
                model_params['n_estimators'] = st.slider("Estimators", 50, 300, 100)
            elif algo == "Neural Network (MLP)":
                model_params['hidden'] = st.text_input("Layers (e.g. 100,50)", "100,50")
                model_params['iter'] = st.slider("Max Iter", 200, 2000, 500)

        else: # Classification
            algo = st.selectbox("Model", ["Logistic Regression", "Random Forest", "SVC (SVM)", "Neural Network (MLP)", "KNN"])
            
            if algo == "Random Forest":
                model_params['n_estimators'] = st.slider("Trees", 10, 500, 100)
            elif algo == "KNN":
                model_params['k'] = st.slider("Neighbors (K)", 1, 20, 5)
            elif algo == "SVC (SVM)":
                model_params['C'] = st.slider("C", 0.1, 10.0, 1.0)
            elif algo == "Neural Network (MLP)":
                model_params['hidden'] = st.text_input("Layers", "100,50")

        st.divider()
        st.markdown("### 🧪 Preprocessing")
        scale_opt = st.selectbox("Scaling", ["None", "StandardScaler", "MinMaxScaler"])
        split_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        
        train_btn = st.button("🚀 TRAIN MODEL")

    with c_res:
        if train_btn:
            if not feats:
                st.error("Please select at least one feature.")
            else:
                X = df[feats]
                y = df[target]
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
                
                # Scaling
                if scale_opt == "StandardScaler":
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                elif scale_opt == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                # Model Init
                model = None
                
                # --- REGRESSION LOGIC ---
                if task == "Regression":
                    if algo == "Linear Regression": model = LinearRegression()
                    elif algo == "Random Forest": model = RandomForestRegressor(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'])
                    elif algo == "SVR (SVM)": model = SVR(C=model_params['C'], kernel=model_params['kernel'])
                    elif algo == "Gradient Boosting": model = GradientBoostingRegressor(learning_rate=model_params['lr'], n_estimators=model_params['n_estimators'])
                    elif algo == "Neural Network (MLP)": 
                        layers = tuple(map(int, model_params['hidden'].split(',')))
                        model = MLPRegressor(hidden_layer_sizes=layers, max_iter=model_params['iter'])
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    
                    st.success("Model Trained Successfully!")
                    m1, m2 = st.columns(2)
                    m1.metric("R² Score", f"{r2:.4f}")
                    m2.metric("MAE", f"{mae:.4f}")
                    
                    # Plot
                    fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, template="plotly_dark", title="Regression Fit")
                    fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)

                # --- CLASSIFICATION LOGIC ---
                else:
                    if algo == "Logistic Regression": model = LogisticRegression()
                    elif algo == "Random Forest": model = RandomForestClassifier(n_estimators=model_params['n_estimators'])
                    elif algo == "KNN": model = KNeighborsClassifier(n_neighbors=model_params['k'])
                    elif algo == "SVC (SVM)": model = SVC(C=model_params['C'])
                    elif algo == "Neural Network (MLP)":
                        layers = tuple(map(int, model_params['hidden'].split(',')))
                        model = MLPClassifier(hidden_layer_sizes=layers)
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, preds)
                    st.success("Model Trained Successfully!")
                    st.metric("Accuracy", f"{acc:.2%}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, preds)
                    fig = px.imshow(cm, text_auto=True, template="plotly_dark", title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)

                # Save Model
                st.session_state.model = model
                
                st.download_button("⬇️ Download Trained Model (.pkl)", download_obj(model, "model"), "model.pkl")

# ==========================================
# 7. NAVIGATION
# ==========================================
with st.sidebar:
    st.title("Navigation")
    
    # Map selection to session state
    nav_map = {"Home": "home", "Data Studio": "data", "EDA Lab": "eda", "Model Forge": "model"}
    rev_map = {v: k for k, v in nav_map.items()}
    
    current = rev_map.get(st.session_state.page, "Home")
    selected = st.radio("Go to:", list(nav_map.keys()), index=list(nav_map.keys()).index(current))
    
    if nav_map[selected] != st.session_state.page:
        st.session_state.page = nav_map[selected]
        st.rerun()

# ==========================================
# 8. ROUTING
# ==========================================
if st.session_state.page == 'home':
    st.title("DATA NEXUS ULTIMATE")
    st.markdown("### The No-Code Analytics Powerhouse")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("### 💾 Data Studio")
        st.markdown("Clean, impute, transform types, and filter your datasets.")
        if st.button("Launch Studio", key="h_d"):
            st.session_state.page = "data"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("### 📊 EDA Lab")
        st.markdown("Univariate, Bivariate, 3D Plotting, and Correlation heatmaps.")
        if st.button("Launch EDA", key="h_e"):
            st.session_state.page = "eda"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Model Forge")
        st.markdown("Train Random Forests, Neural Networks, SVMs with hyperparameter tuning.")
        if st.button("Launch Forge", key="h_m"):
            st.session_state.page = "model"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'data': render_studio()
elif st.session_state.page == 'eda': render_eda()
elif st.session_state.page == 'model': render_forge()
