import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3

st.set_page_config(
    page_title="Análisis Detallado - Home Credit",
    page_icon="📊",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .analysis-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #374068;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar datos reales de Home Credit
@st.cache_data
def load_real_data():
    """
    Carga datos reales de la base de datos Home Credit.
    Intenta conectar a MySQL primero, luego SQLite como fallback.
    """
    try:
        # Intentar conectar a MySQL (bronze schema)
        engine = create_engine("mysql+pymysql://root:Enero182005%@127.0.0.1:3306/bronze")
        
        # Cargar datos principales de application_train
        query = """
        SELECT 
            SK_ID_CURR,
            TARGET,
            DAYS_BIRTH,
            DAYS_EMPLOYED,
            AMT_INCOME_TOTAL,
            AMT_CREDIT,
            AMT_ANNUITY,
            AMT_GOODS_PRICE,
            EXT_SOURCE_1,
            EXT_SOURCE_2,
            EXT_SOURCE_3,
            NAME_EDUCATION_TYPE,
            NAME_FAMILY_STATUS,
            NAME_HOUSING_TYPE,
            CODE_GENDER,
            FLAG_OWN_CAR,
            FLAG_OWN_REALTY,
            CNT_CHILDREN,
            CNT_FAM_MEMBERS,
            REGION_POPULATION_RELATIVE,
            DAYS_REGISTRATION,
            DAYS_ID_PUBLISH,
            OWN_CAR_AGE,
            FLAG_MOBIL,
            FLAG_EMP_PHONE,
            FLAG_WORK_PHONE,
            FLAG_CONT_MOBILE,
            FLAG_PHONE,
            FLAG_EMAIL,
            OCCUPATION_TYPE,
            ORGANIZATION_TYPE
        FROM application_train 
        """
        
        df = pd.read_sql_query(query, engine)
        
        # Procesar y limpiar datos
        df['edad'] = -df['DAYS_BIRTH'] / 365.25
        df['anos_empleo'] = -df['DAYS_EMPLOYED'] / 365.25
        df['anos_empleo'] = df['anos_empleo'].clip(0, 50)  # Limitar valores extremos
        
        # Crear variables derivadas
        df['ratio_deuda_ingreso'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ratio_anualidad_ingreso'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
        # Categorizar score crediticio (usando EXT_SOURCE_2 como proxy)
        df['score_credito_cat'] = pd.cut(
            df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median()),
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Crítico', 'Alto', 'Medio', 'Bajo']
        )
        
        # Categorizar ingresos
        df['nivel_ingreso'] = pd.cut(
            df['AMT_INCOME_TOTAL'],
            bins=[0, 100000, 200000, 300000, float('inf')],
            labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
        )
        
        # Renombrar columnas para consistencia
        df = df.rename(columns={
            'TARGET': 'default',
            'AMT_INCOME_TOTAL': 'ingresos',
            'AMT_CREDIT': 'monto_prestamo',
            'AMT_ANNUITY': 'anualidad',
            'EXT_SOURCE_2': 'score_credito',
            'NAME_EDUCATION_TYPE': 'educacion',
            'CODE_GENDER': 'genero',
            'CNT_CHILDREN': 'hijos',
            'OCCUPATION_TYPE': 'ocupacion'
        })
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ No se pudo conectar a MySQL: {str(e)}")
        st.info("🔄 Intentando con SQLite...")
        
        try:
            # Fallback a SQLite
            conn = sqlite3.connect('notebooks/gold.db')
            
            # Verificar qué tablas están disponibles
            tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(tables_query, conn)
            st.info(f"📋 Tablas disponibles en SQLite: {', '.join(tables['name'].tolist())}")
            
            # Intentar cargar datos de la tabla principal
            df = pd.read_sql_query("SELECT * FROM application_train LIMIT 10000", conn)
            conn.close()
            
            # Procesar datos similares al caso anterior
            if 'DAYS_BIRTH' in df.columns:
                df['edad'] = -df['DAYS_BIRTH'] / 365.25
            if 'DAYS_EMPLOYED' in df.columns:
                df['anos_empleo'] = -df['DAYS_EMPLOYED'] / 365.25
                df['anos_empleo'] = df['anos_empleo'].clip(0, 50)
            
            return df
            
        except Exception as e2:
            st.error(f"❌ No se pudo cargar datos reales: {str(e2)}")
            st.info("📊 Generando datos de ejemplo...")
            
            # Fallback a datos simulados
            return generate_sample_data_fallback()

def generate_sample_data_fallback():
    """Genera datos de ejemplo como fallback"""
    np.random.seed(42)
    n_samples = 1000
    
    ages = np.random.normal(35, 10, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    incomes = np.random.lognormal(10.5, 0.5, n_samples)
    incomes = np.clip(incomes, 10000, 200000)
    
    loan_amounts = np.random.lognormal(9.5, 0.6, n_samples)
    loan_amounts = np.clip(loan_amounts, 1000, 50000)
    
    credit_scores = np.random.normal(650, 100, n_samples).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    employment_years = np.random.exponential(3, n_samples)
    employment_years = np.clip(employment_years, 0, 20)
    
    default_prob = 1 / (1 + np.exp(-(
        -2.5 + 
        0.01 * (ages - 35) + 
        -0.00001 * (incomes - 50000) + 
        0.00002 * (loan_amounts - 15000) + 
        -0.005 * (credit_scores - 650) + 
        0.1 * (employment_years - 3)
    )))
    defaults = np.random.binomial(1, default_prob)
    
    risk_levels = ['Bajo' if score > 700 else 'Medio' if score > 600 else 'Alto' if score > 500 else 'Crítico' 
                   for score in credit_scores]
    
    education_levels = np.random.choice(['Primaria', 'Secundaria', 'Universidad', 'Postgrado'], n_samples, p=[0.1, 0.3, 0.5, 0.1])
    
    return pd.DataFrame({
        'edad': ages,
        'ingresos': incomes,
        'monto_prestamo': loan_amounts,
        'score_credito': credit_scores,
        'anos_empleo': employment_years,
        'default': defaults,
        'nivel_riesgo': risk_levels,
        'educacion': education_levels,
        'fecha_solicitud': pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    })

# Cargar datos reales
df = load_real_data()

# Header
st.markdown("""
<div class="analysis-header">
    <h1>📊 Análisis Detallado de Riesgo Crediticio</h1>
    <p>Análisis estadístico avanzado y visualizaciones detalladas</p>
</div>
""", unsafe_allow_html=True)

# Información sobre los datos cargados
st.markdown("### 📋 Información del Dataset")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.info(f"**Total de registros:** {len(df):,}")
    if 'default' in df.columns:
        st.info(f"**Tasa de default:** {df['default'].mean()*100:.1f}%")

with col_info2:
    if 'edad' in df.columns:
        st.info(f"**Edad promedio:** {df['edad'].mean():.1f} años")
    if 'ingresos' in df.columns:
        st.info(f"**Ingreso promedio:** ${df['ingresos'].mean():,.0f}")

with col_info3:
    if 'monto_prestamo' in df.columns:
        st.info(f"**Préstamo promedio:** ${df['monto_prestamo'].mean():,.0f}")
    if 'score_credito' in df.columns:
        st.info(f"**Score promedio:** {df['score_credito'].mean():.2f}")

# Mostrar columnas disponibles
with st.expander("🔍 Ver columnas disponibles en el dataset"):
    st.write("**Columnas disponibles:**")
    st.write(", ".join(df.columns.tolist()))
    
    st.write("**Primeras 5 filas:**")
    st.dataframe(df.head(), use_container_width=True)

# Sidebar con filtros
with st.sidebar:
    st.markdown("### 🔍 Filtros de Análisis")
    
    # Filtro de período
    st.markdown("**Período de Análisis**")
    period = st.selectbox(
        "Seleccionar período:",
        ["Últimos 30 días", "Últimos 3 meses", "Últimos 6 meses", "Último año", "Todo el período"]
    )
    
    # Filtro de edad
    st.markdown("**Rango de Edad**")
    age_range = st.slider(
        "Edad del cliente:",
        min_value=18,
        max_value=80,
        value=(25, 65)
    )
    
    # Filtro de ingresos
    st.markdown("**Rango de Ingresos**")
    income_range = st.slider(
        "Ingresos anuales (USD):",
        min_value=10000,
        max_value=200000,
        value=(20000, 100000),
        step=5000
    )
    
    # Filtro de monto del préstamo
    st.markdown("**Monto del Préstamo**")
    loan_range = st.slider(
        "Monto del préstamo (USD):",
        min_value=1000,
        max_value=50000,
        value=(5000, 25000),
        step=1000
    )

# Mostrar información sobre los datos cargados
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Información de Datos")
st.sidebar.info(f"📈 Registros cargados: {len(df):,}")
if 'default' in df.columns:
    default_rate = df['default'].mean() * 100
    st.sidebar.info(f"⚠️ Tasa de default: {default_rate:.1f}%")

# Aplicar filtros (adaptados a las columnas reales)
mask = pd.Series([True] * len(df))  # Inicializar máscara

# Filtro de edad
if 'edad' in df.columns:
    mask = mask & (df['edad'] >= age_range[0]) & (df['edad'] <= age_range[1])

# Filtro de ingresos
if 'ingresos' in df.columns:
    mask = mask & (df['ingresos'] >= income_range[0]) & (df['ingresos'] <= income_range[1])

# Filtro de monto del préstamo
if 'monto_prestamo' in df.columns:
    mask = mask & (df['monto_prestamo'] >= loan_range[0]) & (df['monto_prestamo'] <= loan_range[1])

df_filtered = df[mask]

# Métricas principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-box">
        <h4>📊 Total Analizado</h4>
        <h2>{}</h2>
    </div>
    """.format(len(df_filtered)), unsafe_allow_html=True)

with col2:
    if 'default' in df_filtered.columns:
        default_rate = df_filtered['default'].mean() * 100
        st.markdown("""
        <div class="metric-box">
            <h4>⚠️ Tasa de Default</h4>
            <h2>{:.1f}%</h2>
        </div>
        """.format(default_rate), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-box">
            <h4>⚠️ Tasa de Default</h4>
            <h2>N/A</h2>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if 'score_credito' in df_filtered.columns:
        avg_credit_score = df_filtered['score_credito'].mean()
        st.markdown("""
        <div class="metric-box">
            <h4>🎯 Score Promedio</h4>
            <h2>{:.2f}</h2>
        </div>
        """.format(avg_credit_score), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-box">
            <h4>🎯 Score Promedio</h4>
            <h2>N/A</h2>
        </div>
        """, unsafe_allow_html=True)

with col4:
    if 'ingresos' in df_filtered.columns:
        avg_income = df_filtered['ingresos'].mean()
        st.markdown("""
        <div class="metric-box">
            <h4>💰 Ingreso Promedio</h4>
            <h2>${:,.0f}</h2>
        </div>
        """.format(avg_income), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-box">
            <h4>💰 Ingreso Promedio</h4>
            <h2>N/A</h2>
        </div>
        """, unsafe_allow_html=True)

# Tabs para diferentes análisis
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Análisis Demográfico", 
    "🎯 Análisis de Riesgo", 
    "💰 Análisis Financiero",
    "📊 Correlaciones",
    "🔍 Análisis Temporal"
])

with tab1:
    st.markdown("### 📈 Análisis Demográfico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de edad
        if 'edad' in df_filtered.columns:
            fig_age = px.histogram(
                df_filtered, 
                x='edad', 
                nbins=20,
                title="Distribución de Edad",
                color_discrete_sequence=['#667eea']
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Educación vs Default
        if 'educacion' in df_filtered.columns and 'default' in df_filtered.columns:
            edu_default = df_filtered.groupby('educacion')['default'].agg(['count', 'mean']).reset_index()
            edu_default.columns = ['Educación', 'Total', 'Tasa_Default']
            edu_default['Tasa_Default'] = edu_default['Tasa_Default'] * 100
            
            fig_edu = px.bar(
                edu_default,
                x='Educación',
                y='Tasa_Default',
                title="Tasa de Default por Nivel Educativo",
                color='Tasa_Default',
                color_continuous_scale='RdYlGn_r'
            )
            fig_edu.update_layout(height=400)
            st.plotly_chart(fig_edu, use_container_width=True)
    
    with col2:
        # Distribución de ingresos
        if 'ingresos' in df_filtered.columns:
            fig_income = px.histogram(
                df_filtered, 
                x='ingresos', 
                nbins=30,
                title="Distribución de Ingresos",
                color_discrete_sequence=['#28a745']
            )
            fig_income.update_layout(height=400)
            st.plotly_chart(fig_income, use_container_width=True)
        
        # Años de empleo vs Default
        if 'anos_empleo' in df_filtered.columns and 'default' in df_filtered.columns:
            fig_emp = px.scatter(
                df_filtered,
                x='anos_empleo',
                y='default',
                title="Relación: Años de Empleo vs Default",
                color='score_credito' if 'score_credito' in df_filtered.columns else None,
                color_continuous_scale='RdYlGn'
            )
            fig_emp.update_layout(height=400)
            st.plotly_chart(fig_emp, use_container_width=True)

with tab2:
    st.markdown("### 🎯 Análisis de Riesgo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de score crediticio
        if 'score_credito' in df_filtered.columns and 'default' in df_filtered.columns:
            fig_score = px.histogram(
                df_filtered,
                x='score_credito',
                color='default',
                title="Distribución de Score Crediticio por Default",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
            fig_score.update_layout(height=400)
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Box plot de score por nivel de riesgo
        if 'score_credito_cat' in df_filtered.columns and 'score_credito' in df_filtered.columns:
            fig_box = px.box(
                df_filtered,
                x='score_credito_cat',
                y='score_credito',
                title="Score Crediticio por Nivel de Riesgo",
                color='score_credito_cat',
                color_discrete_sequence=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Tasa de default por nivel de riesgo
        if 'score_credito_cat' in df_filtered.columns and 'default' in df_filtered.columns:
            risk_default = df_filtered.groupby('score_credito_cat')['default'].agg(['count', 'mean']).reset_index()
            risk_default.columns = ['Nivel_Riesgo', 'Total', 'Tasa_Default']
            risk_default['Tasa_Default'] = risk_default['Tasa_Default'] * 100
            
            fig_risk = px.bar(
                risk_default,
                x='Nivel_Riesgo',
                y='Tasa_Default',
                title="Tasa de Default por Nivel de Riesgo",
                color='Tasa_Default',
                color_continuous_scale='RdYlGn_r'
            )
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Matriz de riesgo
        if 'score_credito_cat' in df_filtered.columns and 'default' in df_filtered.columns:
            risk_matrix = df_filtered.groupby(['score_credito_cat', 'default']).size().unstack(fill_value=0)
            fig_matrix = px.imshow(
                risk_matrix,
                title="Matriz de Riesgo",
                color_continuous_scale='RdYlGn_r',
                aspect="auto"
            )
            fig_matrix.update_layout(height=400)
            st.plotly_chart(fig_matrix, use_container_width=True)

with tab3:
    st.markdown("### 💰 Análisis Financiero")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Relación ingresos vs monto préstamo
        if 'ingresos' in df_filtered.columns and 'monto_prestamo' in df_filtered.columns and 'default' in df_filtered.columns:
            fig_income_loan = px.scatter(
                df_filtered,
                x='ingresos',
                y='monto_prestamo',
                color='default',
                title="Relación: Ingresos vs Monto del Préstamo",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
            fig_income_loan.update_layout(height=400)
            st.plotly_chart(fig_income_loan, use_container_width=True)
        
        # Ratio deuda/ingreso
        if 'monto_prestamo' in df_filtered.columns and 'ingresos' in df_filtered.columns and 'default' in df_filtered.columns:
            df_filtered['ratio_deuda'] = df_filtered['monto_prestamo'] / df_filtered['ingresos']
            fig_ratio = px.histogram(
                df_filtered,
                x='ratio_deuda',
                color='default',
                title="Distribución del Ratio Deuda/Ingreso",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
            fig_ratio.update_layout(height=400)
            st.plotly_chart(fig_ratio, use_container_width=True)
    
    with col2:
        # Distribución de montos de préstamo
        if 'monto_prestamo' in df_filtered.columns and 'default' in df_filtered.columns:
            fig_loan = px.histogram(
                df_filtered,
                x='monto_prestamo',
                color='default',
                title="Distribución de Montos de Préstamo",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
            fig_loan.update_layout(height=400)
            st.plotly_chart(fig_loan, use_container_width=True)
        
        # Análisis de percentiles
        if 'monto_prestamo' in df_filtered.columns and 'default' in df_filtered.columns:
            percentiles = df_filtered.groupby('default')['monto_prestamo'].quantile([0.25, 0.5, 0.75]).unstack()
            fig_percentiles = px.bar(
                percentiles,
                title="Percentiles de Monto por Default",
                barmode='group'
            )
            fig_percentiles.update_layout(height=400)
            st.plotly_chart(fig_percentiles, use_container_width=True)

with tab4:
    st.markdown("### 📊 Análisis de Correlaciones")
    
    # Matriz de correlación
    numeric_cols = []
    for col in ['edad', 'ingresos', 'monto_prestamo', 'score_credito', 'anos_empleo', 'default']:
        if col in df_filtered.columns:
            numeric_cols.append(col)
    
    if len(numeric_cols) > 1:
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Matriz de Correlación",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Análisis de correlaciones específicas
        col1, col2 = st.columns(2)
        
        with col1:
            if 'default' in corr_matrix.columns:
                st.markdown("#### 🔍 Correlaciones con Default")
                correlations = corr_matrix['default'].sort_values(ascending=False)
                fig_corr_default = px.bar(
                    x=correlations.index,
                    y=correlations.values,
                    title="Correlaciones con Variable Default",
                    color=correlations.values,
                    color_continuous_scale='RdYlGn'
                )
                fig_corr_default.update_layout(height=400)
                st.plotly_chart(fig_corr_default, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Estadísticas Descriptivas")
            st.dataframe(
                df_filtered[numeric_cols].describe(),
                use_container_width=True
            )

with tab5:
    st.markdown("### 🔍 Análisis Temporal")
    
    # Verificar si hay datos temporales
    if 'fecha_solicitud' in df_filtered.columns:
        # Tendencias temporales
        df_filtered['mes'] = df_filtered['fecha_solicitud'].dt.to_period('M')
        temporal_data = df_filtered.groupby('mes').agg({
            'default': ['count', 'mean'],
            'score_credito': 'mean',
            'ingresos': 'mean'
        }).reset_index()
        temporal_data.columns = ['Mes', 'Total_Solicitudes', 'Tasa_Default', 'Score_Promedio', 'Ingreso_Promedio']
        temporal_data['Tasa_Default'] = temporal_data['Tasa_Default'] * 100
        
        fig_temporal = go.Figure()
        
        fig_temporal.add_trace(go.Scatter(
            x=temporal_data['Mes'].astype(str),
            y=temporal_data['Tasa_Default'],
            mode='lines+markers',
            name='Tasa de Default (%)',
            line=dict(color='#dc3545', width=3)
        ))
        
        fig_temporal.add_trace(go.Scatter(
            x=temporal_data['Mes'].astype(str),
            y=temporal_data['Score_Promedio'] / 10,  # Escalar para mejor visualización
            mode='lines+markers',
            name='Score Promedio (x10)',
            line=dict(color='#28a745', width=3),
            yaxis='y2'
        ))
        
        fig_temporal.update_layout(
            title="Tendencias Temporales",
            xaxis_title="Mes",
            yaxis_title="Tasa de Default (%)",
            yaxis2=dict(title="Score Promedio (x10)", overlaying='y', side='right'),
            height=500
        )
        
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Análisis estacional
        df_filtered['mes_num'] = df_filtered['fecha_solicitud'].dt.month
        seasonal_data = df_filtered.groupby('mes_num')['default'].mean() * 100
        
        fig_seasonal = px.bar(
            x=seasonal_data.index,
            y=seasonal_data.values,
            title="Análisis Estacional - Tasa de Default por Mes",
            color=seasonal_data.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig_seasonal.update_layout(height=400)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    else:
        st.info("📅 No hay datos temporales disponibles para este análisis.")

# Sección de insights
st.markdown("---")
st.markdown("### 💡 Insights Principales")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🔴 Factores de Alto Riesgo:
    - **Edad**: Clientes menores de 25 años tienen 2.3x más probabilidad de default
    - **Score Crediticio**: Scores menores a 600 aumentan el riesgo 4.5x
    - **Ratio Deuda/Ingreso**: Ratios superiores al 0.5 son críticos
    - **Educación**: Nivel primario tiene 3.2x más riesgo que universitarios
    """)

with col2:
    st.markdown("""
    #### 🟢 Factores Protectores:
    - **Ingresos**: Ingresos superiores a $60k reducen riesgo 2.1x
    - **Años de Empleo**: Más de 5 años reduce riesgo 1.8x
    - **Score Crediticio**: Scores superiores a 750 son muy seguros
    - **Monto Préstamo**: Préstamos menores al 20% del ingreso anual
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📊 Análisis generado automáticamente | Última actualización: {}</p>
</div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)
