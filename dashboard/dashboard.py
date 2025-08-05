
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4a5568 0%, #2d3748 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4a5568;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header principal con gradiente
st.markdown("""
<div class="main-header">
    <h1>🏦 Home Credit Default Risk Dashboard</h1>
    <p>Análisis integral de riesgo crediticio y predicción de impago</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con navegación
with st.sidebar:
    st.markdown("## 📊 Navegación")
    page = st.selectbox(
        "Selecciona una sección:",
        ["🏠 Dashboard Principal", "📈 Análisis de Datos", "🤖 Modelo ML", "📋 Reportes"]
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    date_range = st.date_input(
        "Rango de fechas",
        value=(datetime.now().date(), datetime.now().date()),
        max_value=datetime.now().date()
    )
    
    st.markdown("---")
    st.markdown("### 📊 Filtros")
    risk_level = st.multiselect(
        "Nivel de Riesgo",
        ["Bajo", "Medio", "Alto", "Crítico"],
        default=["Bajo", "Medio", "Alto", "Crítico"]
    )

# Contenido principal
if page == "🏠 Dashboard Principal":
    # Métricas principales en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total de Préstamos",
            value="125,430",
            delta="+2,340",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="⚠️ En Riesgo",
            value="8,756",
            delta="-156",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="💰 Monto Total",
            value="$45.2M",
            delta="+$1.2M",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="📈 Tasa de Aprobación",
            value="78.5%",
            delta="+2.1%",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Distribución de Riesgo")
        # Datos de ejemplo para el gráfico
        risk_data = pd.DataFrame({
            'Riesgo': ['Bajo', 'Medio', 'Alto', 'Crítico'],
            'Cantidad': [45000, 35000, 25000, 20430],
            'Color': ['#4aa', '#f6ad55', '#fc8181', '#9f7aea']
        })
        
        fig = px.pie(
            risk_data, 
            values='Cantidad', 
            names='Riesgo',
            color_discrete_sequence=risk_data['Color'],
            hole=0.4
        )
        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Tendencia Mensual")
        # Datos de ejemplo para la tendencia
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        trend_data = pd.DataFrame({
            'Fecha': dates,
            'Préstamos': np.random.normal(1000, 200, len(dates)),
            'En Riesgo': np.random.normal(80, 15, len(dates))
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['Fecha'],
            y=trend_data['Préstamos'],
            mode='lines+markers',
            name='Total Préstamos',
            line=dict(color='#4a5568', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=trend_data['Fecha'],
            y=trend_data['En Riesgo'],
            mode='lines+markers',
            name='En Riesgo',
            line=dict(color='#9f7aea', width=3)
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Fecha",
            yaxis_title="Cantidad"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sección de alertas y notificaciones
    st.markdown("---")
    st.markdown("### 🚨 Alertas Recientes")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.info("🔴 **Alta concentración de riesgo** en la región Norte")
        st.warning("🟡 **Aumento del 15%** en solicitudes de alto riesgo")
    
    with alert_col2:
        st.success("🟢 **Mejora del 8%** en la tasa de recuperación")
        st.info("🔵 **Nuevo modelo ML** desplegado exitosamente")

elif page == "📈 Análisis de Datos":
    st.markdown("## 📈 Análisis Detallado de Datos")
    st.write("Aquí encontrarás análisis estadísticos avanzados y visualizaciones detalladas.")
    
    # Placeholder para análisis futuros
    st.info("🚧 Esta sección está en desarrollo. Próximamente: análisis estadísticos avanzados.")

elif page == "🤖 Modelo ML":
    st.markdown("## 🤖 Modelo de Machine Learning")
    st.write("Información sobre el modelo de predicción de riesgo crediticio.")
    
    # Placeholder para información del modelo
    st.info("🚧 Esta sección está en desarrollo. Próximamente: detalles del modelo ML.")

elif page == "📋 Reportes":
    st.markdown("## 📋 Reportes y Documentación")
    st.write("Genera y descarga reportes personalizados.")
    
    # Placeholder para reportes
    st.info("🚧 Esta sección está en desarrollo. Próximamente: generación de reportes.")

# Footer
st.markdown("---")
