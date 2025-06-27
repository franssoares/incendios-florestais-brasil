import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from preprocess import preprocessar_geral
from plotly_plots import (
    plot_incendios_por_ano,
    plot_incendios_por_mes_total,
    plot_incendios_por_estado,
    plot_tendencia_mensal_por_regiao_e_estado,
    plot_media_mensal_incendios,
    plot_tendencia_anual_incendios,
    plot_heatmap_estado_mes
)

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Incêndios Florestais no Brasil",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Função para Carregar Dados ---
@st.cache_data
def load_and_preprocess_data(path: str):
    with st.spinner("Carregando e processando dados..."):
        df = preprocessar_geral(path)
    return df

# --- Página de Introdução ---
def pagina_introducao():
    # CSS personalizado
    st.markdown("""
    <style>
        .header-container {
            position: relative;
            text-align: center;
            margin-bottom: 2rem;
        }
        .header-image {
            width: 100%;
            max-height: 250px;
            object-fit: cover;
            border-radius: 8px;
        }
        .header-title {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 2.2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
            width: 100%;
            padding: 0 1rem;
        }
        @media (max-width: 768px) {
            .header-title {
                font-size: 1.5rem;
            }
        }
        .feature-card {
            background-color: rgba(50,50,50,0.2);
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .data-highlight {
            background-color: rgba(0,0,0,0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header com imagem e título sobreposto
    st.markdown("""
    <div class="header-container">
        <img class="header-image" src="https://f.i.uol.com.br/fotografia/2019/07/21/15637162325d346a8894e9c_1563716232_3x2_md.jpg" alt="Incêndio florestal">
        <div class="header-title"> Análise de Incêndios Florestais no Brasil (1998-2017)</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Incêndio florestal na Amazônia - Fonte: UOL Notícias")

    # Introdução
    st.markdown("""
    ## Bem-vindo ao Dashboard de Análise de Incêndios Florestais
    
    Este dashboard interativo apresenta uma análise abrangente dos dados de focos de incêndio no Brasil entre 
    <span class="data-highlight">1998 e 2017</span>, com base no dataset *Forest Fires in Brazil* do 
    <span class="data-highlight">SNIF (Sistema Nacional de Informações Florestais)</span>.
    """, unsafe_allow_html=True)

    # Seção de objetivos e contexto
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3> Objetivos Principais</h3>
                <ul>
                    <li>Analisar tendências temporais</li>
                    <li>Identificar padrões sazonais</li>
                    <li>Comparar incidência entre estados</li>
                    <li>Fornecer insights para políticas públicas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3> Contexto Importante</h3>
                <ul>
                    <li>Amazônia: maior floresta tropical</li>
                    <li>Ameaça à biodiversidade</li>
                    <li>Impactos globais no clima</li>
                    <li>Dados oficiais do governo</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Seção de dados e navegação
    st.markdown("---")
    st.subheader(" Sobre os Dados e Navegação")
    
    col3, col4 = st.columns(2)
    
    with col3:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h4> Metadados do Dataset</h4>
                <ul>
                    <li><b>Período:</b> 20 anos (1998-2017)</li>
                    <li><b>Fonte:</b> Sistema Nacional de Informações Florestais</li>
                    <li><b>Cobertura:</b> Todos os estados brasileiros</li>
                    <li><b>Variáveis:</b> Focos de incêndio por mês/ano/estado</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h4> Como Utilizar</h4>
                <ul>
                    <li>Use os filtros na barra lateral</li>
                    <li>Explore as diferentes visualizações</li>
                    <li>Clique nos gráficos para interagir</li>
                    <li>Passe o mouse para ver detalhes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Seção de metodologia
    st.markdown("---")
    with st.expander(" Métodologia de Análise (Clique para expandir)", expanded=False):
        st.markdown("""
        ### Pipeline Científico
        
        1. **Coleta de Dados**
           - Obtenção dos dados brutos
           - Verificação de integridade
        
        2. **Processamento**
           - Limpeza e tratamento
           - Transformação de variáveis
           - Criação de indicadores
        
        3. **Análise Exploratória**
           - Estatísticas descritivas
           - Identificação de padrões
           - Visualizações iniciais
        
        4. **Dashboard Interativo**
           - Desenvolvimento de visualizações
           - Implementação de filtros
           - Disponibilização dos resultados
        
        [Acesse o dataset no Kaggle](https://www.kaggle.com/datasets/gustavomodelli/forest-fires-in-brazil)
        """)

    # Rodapé
    st.markdown("---")
    st.caption("""
    Desenvolvido com Python • Streamlit • Plotly | Dados: SNIF/Sistema Nacional de Informações Florestais | Atualizado em 2023  
    [GitHub](https://github.com/franssoares)
    """)

# --- Página de Análise Temporal ---
def analise_temporal(df_final):
    st.title("Análise Temporal dos Incêndios")
    
    st.markdown("""
    Explore as tendências de longo prazo e variações anuais no número de incêndios florestais.
    """)
    
    # Gráficos de tendência anual (mantidos iguais)
    st.subheader("Tendência Anual")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_year = plot_incendios_por_ano(df_final)
        st.plotly_chart(fig_year, use_container_width=True)
    
    with col2:
        df_var = df_final.groupby('year')['number'].sum().reset_index()
        df_var['pct_change'] = df_var['number'].pct_change() * 100
        fig_var = px.bar(df_var, x='year', y='pct_change', 
                        title="Variação Percentual Anual",
                        color='pct_change',
                        color_continuous_scale=px.colors.diverging.RdYlGn_r)
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("---")
    st.subheader("Análise Detalhada por Mês")
    
    # Versão simplificada do heatmap
    heatmap_data = df_final.pivot_table(
        index='year',
        columns='month_num',
        values='number',
        aggfunc='sum'
    ).fillna(0)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Mês", y="Ano", color="Incêndios"),
        x=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
           'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
        color_continuous_scale='Reds',
        aspect="auto"
    )

    fig.update_traces(texttemplate="%{z}", textfont={"size":10})
    
    fig.update_layout(
        yaxis_autorange='reversed',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Insights**:
    - Observe a tendência geral de aumento ou diminuição ao longo dos anos
    - Identifique anos com picos atípicos de incêndios
    - Compare com eventos climáticos ou políticas ambientais relevantes
    - Padrões sazonais consistentes ao longo dos anos
    """)

def distribuicao_geografica(df_final):
    st.title("Distribuição Geográfica dos Incêndios")
    
    st.markdown("""
    Analise a distribuição espacial dos incêndios florestais por estados e regiões do Brasil.
    """)
    
    # Verificação inicial dos dados
    if df_final is None or df_final.empty or len(df_final['state'].unique()) == 0:
        st.warning("Nenhum dado disponível para exibir.")
        return
    
    # Gráficos
    st.subheader("Evolução Anual por Estado")
    fig_state = plot_incendios_por_estado(df_final)
    st.plotly_chart(fig_state, use_container_width=True)
    
    st.subheader("Heatmap: Incêndios por Estado e Mês")
    try:
        fig_heatmap = plot_heatmap_estado_mes(df_final)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except ValueError as e:
        st.error(f"Não foi possível gerar o heatmap: {str(e)}")
    
    st.markdown("""
    **Observações**:
    - Estados da região Norte geralmente apresentam maior incidência
    - Padrões sazonais variam entre regiões
    - Relação entre desmatamento e focos de incêndio
    """)
    
# --- Página de Padrões Sazonais ---
def padroes_sazonais(df_final, selected_region_for_plot):
    st.title(" Padrões Sazonais dos Incêndios")
    
    st.markdown("""
    Identifique os padrões mensais e sazonais na ocorrência de incêndios florestais.
    """)
    
    # Gráficos
    st.subheader("Distribuição Mensal")
    col1, col2 = st.columns(2)
    with col1:
        fig_month = plot_incendios_por_mes_total(df_final)
        st.plotly_chart(fig_month, use_container_width=True) 
    with col2:
        fig_avg = plot_media_mensal_incendios(df_final)
        st.plotly_chart(fig_avg, use_container_width=True)   
    
    st.subheader("Tendência Mensal por Região")
    fig_region = plot_tendencia_mensal_por_regiao_e_estado(df_final, selected_region_for_plot)
    st.plotly_chart(fig_region, use_container_width=True)
    
    st.markdown("""
    **Análise Sazonal**:
    - Períodos de seca geralmente apresentam maior incidência
    - Meses críticos: Agosto a Novembro
    - Variações regionais nos padrões sazonais
    """)

# --- Página de Dados Completos ---
def dados_completos(df_final):
    st.title(" Dados Completos")
    
    st.markdown("""
    Explore o dataset completo utilizado nesta análise.
    """)
    
    st.dataframe(df_final)
    
    st.download_button(
        label="Baixar dados como CSV",
        data=df_final.to_csv().encode('utf-8'),
        file_name='incendios_brasil_filtrado.csv',
        mime='text/csv'
    )

def main():
    # Carregar dados
    CSV_PATH = os.path.join('data', 'amazonfire.csv')
    try:
        df = load_and_preprocess_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de dados '{CSV_PATH}' não encontrado.")
        st.stop()
    
    # Inicializar estados da sessão
    if 'pagina' not in st.session_state:
        st.session_state.pagina = "Introdução"
    if 'selected_states' not in st.session_state:
        st.session_state.selected_states = sorted(df['state'].unique())
    
    # =============================================
    # BARRA LATERAL COM BOTÕES DE NAVEGAÇÃO
    # =============================================
    with st.sidebar:
        # Logo/Cabeçalho
        st.markdown("""
        <div style="text-align:center; margin-bottom:30px;">
            <h2 style="color:#d62728; margin-bottom:0;">🔥 Incêndios Florestais no Brasil</h2>
            <p style="font-size:0.9rem; color:#666;">Análise 1998-2017</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navegação entre páginas
        pages = {
            "Introdução": "",
            "Análise Temporal": "", 
            "Distribuição Geográfica": "",
            "Padrões Sazonais": "",
            "Dados Completos": ""
        }
        
        for page, icon in pages.items():
            if st.button(
                f"{icon} {page}",
                key=f"nav_{page}",
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.pagina = page
        
        # Filtros (exceto para página de Introdução)
        if st.session_state.pagina != "Introdução":
            st.markdown("---")
            with st.expander("FILTROS", expanded=True):
                # Filtro por período
                min_year, max_year = int(df['year'].min()), int(df['year'].max())
                selected_years = st.slider(
                    "Selecione o intervalo de anos:",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    key="year_slider"
                )
                
                # Filtros por localização
                tab1, tab2 = st.tabs(["Estados", "Regiões"])
                
                with tab1:
                    # Usar callback para atualizar os estados selecionados
                    def update_states():
                        st.session_state.selected_states = st.session_state.state_selector
                    
                    selected_states = st.multiselect(
                        "Selecione os estados:",
                        options=sorted(df['state'].unique()),
                        default=st.session_state.selected_states,
                        key="state_selector",
                        on_change=update_states
                    )
                
                with tab2:
                    selected_region = st.selectbox(
                        "Foco em região:",
                        options=["Todas as Regiões"] + sorted(df['regiao'].dropna().unique()),
                        index=0,
                        key="region_selector"
                    )
    
    # =============================================
    # FILTRAGEM DE DADOS
    # =============================================
    if st.session_state.pagina != "Introdução":
        df_filtered = df[
            (df['year'] >= selected_years[0]) & 
            (df['year'] <= selected_years[1]) & 
            (df['state'].isin(st.session_state.selected_states))
        ]
        
        if st.session_state.get("region_selector", "Todas as Regiões") != "Todas as Regiões":
            df_filtered = df_filtered[df_filtered['regiao'] == st.session_state.region_selector]
    else:
        df_filtered = df
    
    # =============================================
    # EXIBIÇÃO DA PÁGINA SELECIONADA
    # =============================================
    if st.session_state.pagina == "Introdução":
        pagina_introducao()
    elif st.session_state.pagina == "Análise Temporal":
        analise_temporal(df_filtered)
    elif st.session_state.pagina == "Distribuição Geográfica":
        distribuicao_geografica(df_filtered)
    elif st.session_state.pagina == "Padrões Sazonais":
        padroes_sazonais(df_filtered, st.session_state.get("region_selector", "Todas as Regiões"))
    else:
        dados_completos(df_filtered)

if __name__ == "__main__":
    main()