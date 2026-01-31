import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="GRIOT AI - DataFlow", layout="wide", page_icon="🔹")

# --- DEFINIÇÃO DA PALETA DE CORES (GRIOT AI) ---
colors = {
    "luster_white": "#F4F1EC",    # Fundo Principal
    "aster_blue": "#9BACD8",      # Detalhes Suaves
    "habanero": "#F98513",        # Botões / Destaque (Laranja)
    "jodhpur_tan": "#DAD1C8",     # Bordas / Secundário
    "deep_royal": "#223382",      # COR PADRÃO DO TEXTO
    "deadly_depths": "#111144"    # Rodapé / Fundo Escuro
}

# --- ESTILIZAÇÃO CSS CUSTOMIZADA (PADRONIZAÇÃO TOTAL) ---
st.markdown(f"""
    <style>
    /* 1. FUNDO GERAL */
    .stApp {{
        background-color: {colors['luster_white']};
    }}

    /* 2. REGRA DE OURO: TUDO NA ÁREA PRINCIPAL É AZUL ESCURO */
    /* Isso força todos os textos, títulos, labels, parágrafos, inputs a serem Deep Royal */
    .main .block-container, 
    .main .block-container div, 
    .main .block-container p, 
    .main .block-container span, 
    .main .block-container label, 
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3, 
    .main .block-container h4, 
    .main .block-container h5, 
    .main .block-container h6,
    .stRadio label, .stCheckbox label, .stSelectbox label, .stMultiSelect label {{
        color: {colors['deep_royal']} !important;
    }}

    /* 3. EXCEÇÃO 1: SIDEBAR (MANTÉM TEXTO BRANCO/CLARO) */
    [data-testid="stSidebar"] {{
        background-color: {colors['deep_royal']};
    }}
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
        color: {colors['luster_white']} !important;
    }}

    /* 4. EXCEÇÃO 2: TABELAS (DATAFRAME) - MANTÉM PRETO PARA LEITURA */
    [data-testid="stDataFrame"] *, [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span {{
        color: black !important;
    }}

    /* 5. EXCEÇÃO 3: BOTÕES (TEXTO BRANCO) */
    .stButton > button {{
        background-color: {colors['habanero']};
        color: white !important;
        border: none;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: #d16e0b;
        color: white !important;
        border: 1px solid white;
    }}
    /* Garante que o texto dentro do botão seja branco (mesmo com a regra global) */
    .stButton > button p {{
        color: white !important;
    }}

    /* 6. EXCEÇÃO 4: ABAS ATIVAS (TEXTO BRANCO) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {colors['jodhpur_tan']};
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {colors['habanero']} !important;
    }}
    /* Texto da aba ativa deve ser branco */
    .stTabs [aria-selected="true"] p {{
        color: white !important;
    }}

    /* 7. RODAPÉ */
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: {colors['deadly_depths']};
        text-align: center;
        padding: 10px;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 14px;
        z-index: 999;
        border-top: 3px solid {colors['habanero']};
    }}
    .footer p {{
        color: {colors['aster_blue']} !important;
        margin: 0;
    }}
    
    .block-container {{
        padding-bottom: 80px;
    }}
    </style>
    """, unsafe_allow_html=True)
st.markdown(f"""
    <style>
    /* --- CORREÇÃO ESPECÍFICA PARA O UPLOAD (CAIXA BRANCA) --- */
    
    /* 1. A zona de drop (fundo e borda) */
    section[data-testid="stFileUploaderDropzone"] {{
        background-color: white !important;
        border: 2px dashed {colors['aster_blue']} !important;
    }}

    /* 2. Todos os textos dentro da zona de drop (Drag and drop..., Limit...) */
    section[data-testid="stFileUploaderDropzone"] div, 
    section[data-testid="stFileUploaderDropzone"] span, 
    section[data-testid="stFileUploaderDropzone"] small {{
        color: {colors['deep_royal']} !important; /* Azul Escuro */
    }}

    /* 3. O Botão "Browse files" específico */
    section[data-testid="stFileUploaderDropzone"] button {{
        background-color: {colors['jodhpur_tan']} !important; /* Bege Jodhpur */
        color: {colors['deep_royal']} !important; /* Texto Azul Escuro */
        border: 1px solid {colors['deep_royal']} !important;
        font-weight: bold;
    }}
    
    /* 4. Efeito Hover no botão Browse */
    section[data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: {colors['aster_blue']} !important;
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            try:
                return pd.read_csv(file)
            except UnicodeDecodeError:
                file.seek(0)
                try:
                    return pd.read_csv(file, encoding='latin-1', sep=';')
                except:
                    file.seek(0)
                    return pd.read_csv(file, encoding='latin-1')
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            return pd.read_excel(file)
        elif file.name.endswith('.parquet'):
            return pd.read_parquet(file)
    except Exception as e:
        st.error(f"Erro crítico: {e}")
        return None

# --- ESTADO ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None

# --- SIDEBAR (IDENTIDADE VISUAL) ---
with st.sidebar:
    # Tenta carregar logo em vários formatos
    if os.path.exists("logo_griot.png"):
        st.image("logo_griot.png", use_container_width=True)
    elif os.path.exists("logo_griot.jpeg"):
        st.image("logo_griot.jpeg", use_container_width=True)
    elif os.path.exists("WhatsApp Image 2026-01-08 at 23.46.19.jpeg"):
        st.image("WhatsApp Image 2026-01-08 at 23.46.19.jpeg", use_container_width=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: white;'>GRIOT AI</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📂 Central de Dados")
    uploaded_file = st.file_uploader("Carregar Arquivo", type=['csv', 'xlsx', 'parquet'])
    
    if uploaded_file is not None:
        if st.session_state.df is None: 
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.success(" Carregado!")
                st.rerun()

    if st.session_state.df is not None:
        st.markdown("---")
        if st.button("🔄 Reiniciar Dataset"):
            st.session_state.df = st.session_state.df_original.copy()
            st.rerun()
        
        st.info(f" Linhas: {st.session_state.df.shape[0]}\n Colunas: {st.session_state.df.shape[1]}")

# --- APP PRINCIPAL ---
st.markdown(f"<h1 style='color:{colors['deep_royal']}'>DataFlow Studio <span style='font-size:0.5em; color:{colors['habanero']}'>by Griot AI</span></h1>", unsafe_allow_html=True)

if st.session_state.df is not None:
    df = st.session_state.df

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Overview", " Limpeza", " Visual", " AutoML", " Exportar"
    ])

    # --- ABA 1: OVERVIEW ---
    with tab1:
        st.subheader("Raio-X dos Dados")
        
        c1, c2, c3, c4 = st.columns(4)
        total = df.size
        missing = df.isnull().sum().sum()
        
        c1.metric("Variáveis", df.shape[1])
        c2.metric("Observações", df.shape[0])
        c3.metric("Nulos", f"{missing} ({(missing/total)*100:.1f}%)")
        c4.metric("Duplicatas", df.duplicated().sum())
        
        st.divider()
        st.dataframe(df.head(), use_container_width=True)
        
        st.write("##### Saúde das Colunas")
        dtypes = pd.DataFrame({'Tipo': df.dtypes, 'Nulos': df.isnull().sum(), '% Nulos': (df.isnull().sum()/len(df))*100})
        st.dataframe(dtypes.style.background_gradient(cmap='Oranges', subset=['% Nulos']), use_container_width=True)

    # --- ABA 2: LIMPEZA ---
    with tab2:
        st.subheader("Tratamento de Dados")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Ferramenta:")
            option = st.radio("Ação:", ["Excluir Colunas", "Tratar Nulos", "Remover Duplicatas", "Renomear"], label_visibility="collapsed")
        with c2:
            if option == "Excluir Colunas":
                cols = st.multiselect("Colunas:", df.columns)
                if st.button("🗑️ Excluir"):
                    st.session_state.df = df.drop(columns=cols)
                    st.rerun()
            elif option == "Tratar Nulos":
                cols_null = df.columns[df.isnull().any()]
                if len(cols_null) > 0:
                    col = st.selectbox("Coluna:", cols_null)
                    method = st.selectbox("Ação:", ["Remover Linhas", "Média", "Mediana", "Zero/Valor"])
                    if st.button("Aplicar"):
                        if method == "Remover Linhas": st.session_state.df = df.dropna(subset=[col])
                        elif method == "Média": st.session_state.df[col] = df[col].fillna(df[col].mean())
                        elif method == "Mediana": st.session_state.df[col] = df[col].fillna(df[col].median())
                        elif method == "Zero/Valor": st.session_state.df[col] = df[col].fillna(0)
                        st.rerun()
                else:
                    st.success("Sem dados nulos!")
            elif option == "Remover Duplicatas":
                if st.button("Aplicar"):
                    st.session_state.df = df.drop_duplicates()
                    st.rerun()
            elif option == "Renomear":
                col = st.selectbox("Coluna:", df.columns)
                novo = st.text_input("Novo nome:")
                if st.button("Renomear"):
                    st.session_state.df = df.rename(columns={col: novo})
                    st.rerun()
                    
        st.dataframe(st.session_state.df.head(3), use_container_width=True)

    # --- ABA 3: VISUAL ---
    with tab3:
        st.subheader("Análise Gráfica")
        tipo = st.selectbox("Visualização:", ["Histograma", "Dispersão", "Correlação"])
        if tipo == "Histograma":
            cx = st.selectbox("Eixo X:", df.select_dtypes(include='number').columns)
            if cx: st.plotly_chart(px.histogram(df, x=cx, color_discrete_sequence=[colors['habanero']]), use_container_width=True)
        elif tipo == "Dispersão":
            c1, c2 = st.columns(2)
            cx = c1.selectbox("Eixo X:", df.select_dtypes(include='number').columns, key='vx')
            cy = c2.selectbox("Eixo Y:", df.select_dtypes(include='number').columns, key='vy')
            if cx and cy: st.plotly_chart(px.scatter(df, x=cx, y=cy, color_discrete_sequence=[colors['deep_royal']]), use_container_width=True)
        elif tipo == "Correlação":
            num = df.select_dtypes(include='number')
            if not num.empty: st.plotly_chart(px.imshow(num.corr(), text_auto=True, color_continuous_scale='RdBu'), use_container_width=True)

    # --- ABA 4: AUTOML ---
    with tab4:
        st.subheader(" AutoML GRIOT")
        c1, c2 = st.columns([1, 2])
        with c1:
            target = st.selectbox("Alvo (Target):", df.columns)
            feats = st.multiselect("Features:", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
            btn = st.button(" Treinar Modelo")
        with c2:
            if btn:
                df_ml = df[feats + [target]].dropna()
                X = pd.get_dummies(df_ml[feats], drop_first=True)
                y = df_ml[target]
                
                # Regressão ou Classificação?
                is_class = False
                if y.dtype == 'object' or y.nunique() < 20:
                    is_class = True
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                if is_class:
                    mdl = RandomForestClassifier()
                    mdl.fit(X_train, y_train)
                    acc = accuracy_score(y_test, mdl.predict(X_test))
                    st.success(f"Acurácia: {acc:.1%}")
                else:
                    mdl = RandomForestRegressor()
                    mdl.fit(X_train, y_train)
                    r2 = r2_score(y_test, mdl.predict(X_test))
                    st.success(f"R² Score: {r2:.2f}")

    # --- ABA 5: EXPORTAR ---
    with tab5:
        st.subheader("Exportar")
        st.download_button(" Baixar Dataset Limpo", data=convert_df(df), file_name='griot_dataflow.csv', mime='text/csv')

else:
    # TELA DE WELCOME
    st.markdown(f"""
    <div style="text-align: center; padding: 50px; background-color: white; border-radius: 10px; border: 1px solid {colors['aster_blue']};">
        <h1 style="color: {colors['deep_royal']};">Bem-vindo ao DataFlow Studio</h1>
        <p style="color: {colors['habanero']}; font-weight: bold;">POWERED BY GRIOT AI</p>
        <p style="color: gray;">Carregue seus dados na barra lateral para iniciar a análise.</p>
    </div>
    """, unsafe_allow_html=True)

# --- RODAPÉ FIXO ---
st.markdown("""
    <div class="footer">
        <p>Desenvolvido por <b>GRIOT AI</b> @2026 | DataFlow Studio v5.0</p>
    </div>
    """, unsafe_allow_html=True)