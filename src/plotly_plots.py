#src/plotly_plots.py
import plotly.express as px
import pandas as pd

def plot_incendios_por_ano(df: pd.DataFrame):
    """
    Cria um gráfico de linha interativo do número total de incêndios por ano.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas 'year' e 'number'.

    Returns:
        plotly.graph_objects.Figure: Objeto de figura Plotly.
    """
    df_total_ano = df.groupby("year", as_index=False)["number"].sum()
    fig = px.line(df_total_ano, x='year', y='number',
                  title="Número Total de Incêndios por Ano",
                  labels={'year': 'Ano', 'number': 'Número de Incêndios'},
                  markers=True)
    #diciona funcionalidade de hover unificada para melhor interatividade
    fig.update_layout(hovermode="x unified")
    return fig

def plot_incendios_por_mes_total(df: pd.DataFrame):
    """
    Cria um gráfico de barras interativo do número total de queimadas por mês.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas 'month_num' e 'number'.

    Returns:
        plotly.graph_objects.Figure: Objeto de figura Plotly.
    """
    df_total_mes = df.groupby("month_num", as_index=False)["number"].sum()
    fig = px.bar(df_total_mes, x='month_num', y='number',
                 title="Número Total de Queimadas por Mês",
                 labels={'month_num': 'Mês', 'number': 'Quantidade de Queimadas'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    # configura os rótulos do eixo x para mostrar os nomes abreviados dos meses
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    return fig

def plot_incendios_por_estado(df: pd.DataFrame):
    """
    Cria um gráfico de barras horizontal animado da evolução de incêndios por estado ao longo dos anos.
    
    Args:
        df (pd.DataFrame): DataFrame contendo as colunas 'state', 'number' e 'year'.
        
    Returns:
        plotly.graph_objects.Figure: Objeto de figura Plotly animado.
    """
    # Agrupa por ano e estado
    df_anim = df.groupby(['year', 'state'])['number'].sum().reset_index()
    
    # Ordena por ano e número de incêndios (para animação fluida)
    df_anim = df_anim.sort_values(['year', 'number'], ascending=[True, False])
    
    # Cria o gráfico animado
    fig = px.bar(
        df_anim,
        x='number',
        y='state',
        animation_frame='year',
        orientation='h',
        title="Evolução Anual de Incêndios por Estado (1998–2017)",
        labels={'number': 'Número de Incêndios', 'state': 'Estado', 'year': 'Ano'},
        color='state',
        color_discrete_sequence=px.colors.sequential.Reds_r,
        range_x=[0, df_anim['number'].max() * 1.1] 
    )
    
    # Configurações de layout e animação (com tempos ajustados)
    fig.update_layout(
        showlegend=False,
        height=600,
        margin=dict(l=100, r=50, t=80, b=50),
        xaxis_title="Número de Incêndios",
        yaxis_title="Estado",
        yaxis={'categoryorder': 'total ascending'},
        transition={'duration': 2000},
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 2000, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 1500, 'easing': 'linear'}
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 70},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    # Suaviza a transição entre frames
    for frame in fig.frames:
        frame.layout.update(transition={'duration': 5000})  
    
    # Adiciona labels nas barras
    fig.update_traces(
        texttemplate='%{x:.0f}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Incêndios: %{x:,}<extra></extra>'
    )
    
    # Configurações do slider (com transição mais lenta)
    if len(fig.layout.sliders) > 0:
        fig.layout.sliders[0].currentvalue = {
            'prefix': 'Ano: ',
            'font': {'size': 14},
            'xanchor': 'right'
        }
        fig.layout.sliders[0].transition = {'duration': 2000, 'easing': 'linear'}  
    
    return fig

def plot_tendencia_mensal_por_regiao_e_estado(df: pd.DataFrame, selected_region: str = None):
    """
    Cria um gráfico de linha interativo da média mensal de incêndios por estado,
    com opção de filtrar por região.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas 'regiao', 'state', 'month_num' e 'number'.
        selected_region (str, optional): A região selecionada para filtrar.
                                         Se None ou "Todas as Regiões", mostra todos.

    Returns:
        plotly.graph_objects.Figure: Objeto de figura Plotly.
    """
    df_media = df.groupby(['regiao', 'state', 'month_num'])['number'].mean().reset_index()

    if selected_region and selected_region != "Todas as Regiões":
        df_regiao = df_media[df_media['regiao'] == selected_region]
        title = f"Média Mensal de Incêndios - Região {selected_region}"
    else:
        df_regiao = df_media
        title = "Média Mensal de Incêndios por Região e Estado"

    fig = px.line(df_regiao, x='month_num', y='number', color='state',
                  line_group='state', hover_name='state',
                  title=title,
                  labels={'month_num': 'Mês', 'number': 'Média de Incêndios'})
    # Configura os rótulos do eixo X para mostrar os nomes abreviados dos meses
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    fig.update_layout(hovermode="x unified")
    return fig

def plot_media_mensal_incendios(df: pd.DataFrame):
    """
    Cria um gráfico de linha interativo da média mensal de incêndios ao longo de todos os anos.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas 'month_num' e 'number'.

    Returns:
        plotly.graph_objects.Figure: Objeto de figura Plotly.
    """
    media_mes = df.groupby('month_num')['number'].mean().reset_index()
    fig = px.line(media_mes, x='month_num', y='number',
                  title="Média Mensal do Número de Incêndios (Todos os Anos)",
                  labels={'month_num': 'Mês', 'number': 'Número Médio de Incêndios'},
                  markers=True, color_discrete_sequence=['blue'])
    #Configura os rótulos do eixo X para mostrar os nomes abreviados dos meses
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    fig.update_layout(hovermode="x unified")
    return fig

def plot_tendencia_anual_incendios(df: pd.DataFrame):
    """
    Cria um gráfico de linha interativo da tendência anual do número total de incêndios.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas 'year' e 'number'.

    Returns:
        plotly.graph_objects.Figure: Objeto de figura Plotly.
    """
    # Agrupa por ano e soma para ter o total anual
    df_total_ano = df.groupby('year')['number'].sum().reset_index()
    fig = px.line(df_total_ano, x='year', y='number',
                  title="Tendência Anual do Número de Incêndios",
                  labels={'year': 'Ano', 'number': 'Número de Incêndios'},
                  markers=True, color_discrete_sequence=['green'])
    fig.update_layout(hovermode="x unified")
    return fig

def plot_heatmap_estado_mes(df):
    # Verifica se há dados
    if df is None or df.empty or len(df['state'].unique()) == 0:
        raise ValueError("Dados insuficientes para gerar o heatmap")
    
    # Mapeamento de meses para números
    month_map = {
        'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4,
        'Maio': 5, 'Junho': 6, 'Julho': 7, 'Agosto': 8,
        'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12
    }
    
    # Cria uma cópia para não modificar o DataFrame original
    df = df.copy()
    
    # Converte nomes de meses para números
    df['month_num'] = df['month'].map(month_map)
    
    # Calcula a média de incêndios por estado e mês
    df_grouped = df.groupby(['state', 'month_num'])['number'].mean().reset_index()
    
    # Cria um DataFrame completo com todas combinações de estado e mês
    complete_index = pd.MultiIndex.from_product(
        [sorted(df['state'].unique()), range(1, 13)],
        names=['state', 'month_num']
    )
    
    df_complete = pd.DataFrame(index=complete_index).reset_index()
    df_complete = pd.merge(df_complete, df_grouped, on=['state', 'month_num'], how='left')
    df_complete['number'] = df_complete['number'].fillna(0)
    
    # Pivot para formato de matriz
    pivot = df_complete.pivot(index='state', columns='month_num', values='number')
    pivot = pivot[list(range(1, 13))]  # Garante ordem correta dos meses
    
    # Nomes dos meses para exibição
    month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    # Paleta de cores personalizada
    colorscale = [
        [0.0, 'white'],
        [0.2, "#ffadad"],
        [0.4, "#f86262"],
        [0.6, "#ff1a1a"],
        [0.8, "#b30000"],
        [1.0, "#550000"]
    ]
    
    # Cria o heatmap
    fig = px.imshow(
        pivot,
        labels=dict(x="Mês", y="Estado", color="Incêndios Médios"),
        x=month_names,
        color_continuous_scale=colorscale,
        aspect="auto",
        text_auto=".0f",  # Mostra valores inteiros
        zmin=0
    )
    
    # Calcula altura baseada no número de estados
    num_states = len(pivot.index)
    height = max(400, num_states * 30)  # Altura mínima de 400px
    
    # Configurações de layout
    fig.update_layout(
        title="Média de Incêndios por Estado e Mês",
        xaxis_title="Mês",
        yaxis_title="Estado",
        height=height,
        width=800,
        margin=dict(l=100, r=50, t=80, b=50),
        font=dict(size=12),
        coloraxis_colorbar=dict(
            title="Média",
            thickness=15,
            len=0.6
        )
    )
    
    # Configurações para células quadradas e texto visível
    fig.update_traces(
        xgap=1,
        ygap=1,
        textfont=dict(
            size=11,
            color='black'  # Texto sempre branco
        ),
        hovertemplate=(
            "<b>Estado:</b> %{y}<br>"
            "<b>Mês:</b> %{x}<br>"
            "<b>Média de incêndios:</b> %{z:.1f}<br>"
            "<extra></extra>"
        )
    )
    
    return fig
#---------------------------------------------------------

