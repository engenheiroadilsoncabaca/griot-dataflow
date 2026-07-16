import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sqlite3
import streamlit_authenticator as stauth
import bcrypt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

# ==========================================
# 1. CONFIGURAÇÃO INICIAL E ESTILOS
# ==========================================
st.set_page_config(page_title="DataFlow Studio", page_icon="", layout="wide")

# CSS da GRIOT AI
colors = {
    'luster_white': '#F4F1EC', 'deep_royal': '#223382',
    'habanero': '#F98513', 'aster_blue': '#798ECA', 'deadly_depths': '#15172C'
}
st.markdown(f"""
    <style>
    .stApp {{ background-color: {colors['luster_white']}; }}
    h1, h2, h3, h4, h5, h6, .stHeadingContainer {{ color: {colors['deep_royal']} !important; }}
    [data-testid="stMetricLabel"] {{ color: {colors['deep_royal']} !important; font-weight: bold; }}
    [data-testid="stMetricValue"] {{ color: {colors['deep_royal']} !important; }}
    .stMarkdown p, .stMarkdown li {{ color: {colors['deep_royal']} !important; }}
    [data-testid="stSidebar"] {{ background-color: {colors['deep_royal']}; }}
    [data-testid="stSidebar"] * {{ color: {colors['luster_white']} !important; }}
    .stButton>button {{ background-color: {colors['habanero']}; color: white !important; border: none; }}
    .stTabs [aria-selected="true"] {{ background-color: {colors['habanero']} !important; }}
    .stTabs [aria-selected="true"] p {{ color: white !important; }}
    .block-container {{ padding-bottom: 80px; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BANCO DE DADOS DE USUÁRIOS (Contas)
# ==========================================
# Cria/Conecta ao banco local para salvar os cadastros
conn_users = sqlite3.connect('users.db', check_same_thread=False)
c = conn_users.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, name TEXT, email TEXT, password TEXT)''')
conn_users.commit()

# Função para buscar usuários do banco
def get_users():
    c.execute("SELECT * FROM users")
    users_db = c.fetchall()
    credentials = {"usernames": {}}
    for user in users_db:
        credentials["usernames"][user[0]] = {"name": user[1], "email": user[2], "password": user[3]}
    return credentials

# ==========================================
# 3. SISTEMA DE LOGIN E CADASTRO
# ==========================================
credentials = get_users()

authenticator = stauth.Authenticate(
    credentials,
    "dataflow_cookie",
    "griot_key_super_secreta_e_segura_2026",
    cookie_expiry_days=30
)

# Verifica se o usuário está logado
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

if not st.session_state['authentication_status']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center;'>🔐 DataFlow Studio</h1>", unsafe_allow_html=True)
        
        tab_login, tab_cadastro = st.tabs(["Entrar", "Criar Conta"])
        
        with tab_login:
            # Formulário nativo do Authenticator (Versão mais recente)
            authenticator.login(location="main")
            
            if st.session_state["authentication_status"] is False:
                st.error("Usuário ou senha incorretos")
            elif st.session_state["authentication_status"] is None:
                st.warning("Insira suas credenciais")

        with tab_cadastro:
            st.subheader("Novo Usuário")
            new_user = st.text_input("Usuário (Login)")
            new_name = st.text_input("Nome Completo")
            new_email = st.text_input("E-mail")
            new_password = st.text_input("Senha", type="password")
            
            if st.button("Cadastrar e Criptografar"):
                if new_user and new_password:
                    try:
                        # Criptografa a senha antes de salvar
                        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        c.execute("INSERT INTO users (username, name, email, password) VALUES (?, ?, ?, ?)",
                                  (new_user, new_name, new_email, hashed_password))
                        conn_users.commit()
                        st.success("Conta criada com sucesso! Você já pode fazer login na aba 'Entrar'.")
                    except sqlite3.IntegrityError:
                        st.error("Este nome de usuário já existe. Tente outro.")
                else:
                    st.warning("Preencha todos os campos.")

# ==========================================
# O APLICATIVO REAL (PÓS-LOGIN)
# ==========================================
if st.session_state['authentication_status']:
    
    authenticator.logout(button_name='Sair (Logout)', location='sidebar')
    st.sidebar.title(f"Bem-vindo, {st.session_state['name']}!")
    
    st.markdown(f"<h1>DataFlow Studio <span style='color:{colors['habanero']}; font-size: 0.6em;'>by Griot AI</span></h1>", unsafe_allow_html=True)

    # Armazena o DataFrame na sessão para não perder os dados ao trocar de abas
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    # ==========================================
    # 4. ENTRADA DE DADOS (Arquivo vs SQL)
    # ==========================================
    st.write("### Fonte de Dados")
    tab_arq, tab_sql = st.tabs(["📂 Upload de Arquivo", "🗄️ Conectar Banco SQL"])
    
    with tab_arq:
        uploaded_file = st.file_uploader("Suba seu arquivo (CSV ou Excel)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state['df'] = pd.read_csv(uploaded_file)
                else:
                    st.session_state['df'] = pd.read_excel(uploaded_file)
                st.success("Arquivo carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

    with tab_sql:
        st.info("Conecte-se diretamente ao banco de dados da sua empresa.")
        c1, c2 = st.columns(2)
        db_type = c1.selectbox("Tipo de Banco", ["mysql+pymysql", "postgresql", "sqlite"], key="sql_tipo")
        db_host = c2.text_input("Host (ex: localhost...)", placeholder="127.0.0.1", key="sql_host")
        db_port = c1.text_input("Porta (MySQL: 3306...)", placeholder="3306", key="sql_porta")
        db_user = c2.text_input("Usuário", key="sql_user")
        db_pass = c1.text_input("Senha", type="password", key="sql_pass")
        db_name = c2.text_input("Nome do Banco", key="sql_db")
        
        query = st.text_area("Consulta (Query)", "SELECT * FROM sua_tabela LIMIT 5000")
        
        if st.button("Executar Query e Importar"):
            try:
                with st.spinner("Conectando ao servidor..."):
                    # Monta a String de Conexão
                    if db_type == "sqlite":
                        uri = f"sqlite:///{db_name}" # SQLite é local
                    else:
                        uri = f"{db_type}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
                    
                    engine = create_engine(uri)
                    st.session_state['df'] = pd.read_sql(query, engine)
                    st.success("Dados importados com sucesso via SQL!")
            except Exception as e:
                st.error(f"Erro de conexão: {e}")

    st.markdown("---")

    # ==========================================
    # 5. ANÁLISE E MACHINE LEARNING
    # ==========================================
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        t1, t2, t3, t4 = st.tabs([" Overview", " Limpeza", " Visual", " AutoML"])

        with t1:
            st.subheader("Raio-X dos Dados")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Variáveis", df.shape[1])
            c2.metric("Observações", df.shape[0])
            nulos_totais = df.isnull().sum().sum()
            c3.metric("Nulos", f"{nulos_totais}")
            c4.metric("Duplicatas", df.duplicated().sum())
            st.dataframe(df.head(10), use_container_width=True)

        with t2:
            st.subheader(" Limpeza Expressa")
            c1, c2, c3 = st.columns(3)
            if c1.button("Remover Duplicatas"):
                df.drop_duplicates(inplace=True)
                st.session_state['df'] = df
                st.success("Duplicatas removidas!")
                st.rerun()
            if c2.button("Remover Nulos"):
                df.dropna(inplace=True)
                st.session_state['df'] = df
                st.success("Linhas nulas removidas!")
                st.rerun()
            if c3.button("Preencher Nulos (Zero/Vazio)"):
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col].fillna("Vazio", inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
                st.session_state['df'] = df
                st.success("Nulos preenchidos!")
                st.rerun()

        with t3:
            st.subheader(" Gráficos Interativos")
            tipo = st.selectbox("Tipo", ["Dispersão (Scatter)", "Barras", "Boxplot"])
            c1, c2, c3 = st.columns(3)
            col_x = c1.selectbox("Eixo X", df.columns)
            col_y = c2.selectbox("Eixo Y", df.columns)
            col_cor = c3.selectbox("Cor (Opcional)", ["Nenhum"] + list(df.columns))
            cor = None if col_cor == "Nenhum" else col_cor

            if st.button("Gerar Gráfico"):
                try:
                    if tipo == "Dispersão (Scatter)": fig = px.scatter(df, x=col_x, y=col_y, color=cor)
                    elif tipo == "Barras": fig = px.bar(df, x=col_x, y=col_y, color=cor)
                    elif tipo == "Boxplot": fig = px.box(df, x=col_x, y=col_y, color=cor)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("Erro ao gerar gráfico. Verifique o tipo das colunas.")

        with t4:
            st.subheader(" AutoML Machine Learning")
            c1, c2 = st.columns([1,2])
            with c1:
                target = st.selectbox("O que prever? (Target)", df.columns)
                features = st.multiselect("Features", [c for c in df.columns if c != target])
                split = st.slider("Tamanho Teste (%)", 10, 50, 20)
                btn_train = st.button("Treinar Modelo")

            with c2:
                if btn_train and features:
                    try:
                        df_ml = df[features + [target]].dropna()
                        
                        # Proteção de Memória
                        high_cardinality = [col for col in features if df_ml[col].dtype == 'object' and df_ml[col].nunique() > 50]
                        if high_cardinality:
                            st.error(f"Remova colunas com muitos textos únicos para não travar: {high_cardinality}")
                        else:
                            X = pd.get_dummies(df_ml[features], drop_first=True)
                            y = df_ml[target]
                            
                            is_class = (y.dtype == 'object' or y.nunique() < 20)
                            if is_class: y = LabelEncoder().fit_transform(y)
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split/100, random_state=42)
                            
                            with st.spinner("Treinando..."):
                                if is_class:
                                    model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
                                    acc = accuracy_score(y_test, model.predict(X_test))
                                    st.success(f"Classificação - Acurácia: {acc:.1%}")
                                else:
                                    model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)
                                    st.success(f"Regressão - R²: {r2_score(y_test, model.predict(X_test)):.2f}")
                    except Exception as e:
                        st.error(f"Erro no treinamento: {e}")