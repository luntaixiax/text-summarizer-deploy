import os
from typing import List
import pandas as pd
import numpy as np
from itertools import cycle
from datetime import datetime, timedelta
from bokeh.plotting import figure, show, Figure
from bokeh.models import ColumnDataSource, HoverTool, TableColumn, DataTable, \
    Range1d, LinearAxis, Legend, CustomJS, Slider, \
    Column, ColorBar, BasicTicker, Row, Select, Panel, Tabs, DateFormatter, Span
from bokeh.layouts import column, row, layout, gridplot
from bokeh.palettes import Spectral, Dark2_5, Category20_8

def table_nonedit_general(df:pd.DataFrame, index_col:str = None, 
        dt_columns: List[str] = None, index_position: int = None):
    df = df.copy()
    if index_col is not None:
        df = df.set_index(index_col)
    dt_columns = [] if dt_columns is None else dt_columns
    for col in dt_columns:
        df[col] = pd.to_datetime(df[col])
    
    source = ColumnDataSource(df)
    columns = [TableColumn(field = col, title = col, formatter = DateFormatter()) for col in dt_columns]
    columns.extend([TableColumn(field = col, title = col) for col in df.columns if col not in dt_columns])
    data_table = DataTable(source = source, columns = columns, editable = True, 
                           index_position = index_position, sizing_mode = "stretch_both")
    return data_table

def chart_hist_stat(hist_stat: pd.DataFrame, freq:str = 'h', size: tuple = (1200, 400), title: str = 'historical stat') -> Figure:
    hist_stat['prediction_dt'] = pd.to_datetime(hist_stat['prediction_dt'])
    hist_stat_unpivot = (
        hist_stat
        .pivot_table(
            values = 'num_record', 
            index = 'prediction_dt', 
            columns = 'model_source', 
            aggfunc = 'sum'
        )
        .fillna(0)
        .reset_index()
    )
    labels = pd.unique(values = hist_stat['model_source']).tolist()
    platte = cycle(Dark2_5)
    colors = [next(platte) for l in labels]
    source = ColumnDataSource(hist_stat_unpivot)

    width = {
        "Hour" : dict(minutes = 45),
        "Day" : dict(hours = 20),
        "Month" : dict(days = 25)
    }.get(freq)

    fig = figure(
        plot_width = size[0], plot_height = size[1], title = title,
        x_axis_label = 'datetime', y_axis_label = '# of predictions',
        x_axis_type = 'datetime'
    )
    fig.add_layout(Legend(orientation = 'horizontal'), 'above')
    fig.vbar_stack(
        labels,
        width = timedelta(**width),
        x = 'prediction_dt',
        source = source,
        legend_label = labels,
        color = colors
    )

    hovers = [(f, f"@{f}") for f in labels]
    fig.add_tools(
        HoverTool(tooltips = [
            ('date', '@prediction_dt{%F}'),
            *hovers
        ],
        formatters = {'@prediction_dt': 'datetime'}),
    )
    return fig

def chart_hist_score(hist_score: pd.DataFrame, metric: str, freq:str = 'h', size: tuple = (1200, 400), title: str = 'historical scores') -> Figure:
    colors = cycle(Category20_8)
    hist_score['prediction_dt'] = pd.to_datetime(hist_score['prediction_dt'])
    hist_score[f'-sigma_{metric}'] = hist_score[f'score_{metric}'] - hist_score[f'std_{metric}']
    hist_score[f'+sigma_{metric}'] = hist_score[f'score_{metric}'] + hist_score[f'std_{metric}']

    source = ColumnDataSource(hist_score)
    fig = figure(
        plot_width = size[0], plot_height = size[1], title = title,
        x_axis_label = 'datetime', y_axis_label = metric,
        x_axis_type = 'datetime'
    )
    fig.add_layout(Legend(orientation = 'horizontal'), 'above')

    fig.varea(
        source = source, x = 'prediction_dt', y1 = f'min_{metric}', y2 = f'max_{metric}',
        alpha = 0.4, fill_color = '#cdd1d5', legend_label = "min_max"
    )

    fig.varea(
        source = source, x = 'prediction_dt', y1 = f'-sigma_{metric}', y2 = f'+sigma_{metric}',
        alpha = 0.4, fill_color = '#7dc1dc', legend_label = "±σ"
    )
    fig.line(
        'prediction_dt', f'min_{metric}', source = source,
        width = 2, legend_label = 'min', color = '#46626e',
        line_dash = "4 4"
    )
    fig.line(
        'prediction_dt', f'score_{metric}', source = source,
        width = 3, legend_label = 'avg', color = '#129fd8'
    )
    fig.circle(
        'prediction_dt', f'score_{metric}', source = source,
        color = '#3d8dad', fill_color = 'white', size = 10
    )
    fig.line(
        'prediction_dt', f'max_{metric}', source = source,
        width = 2, legend_label = 'max', color = '#46626e',
        line_dash = "4 4"
    )
    fig.add_tools(
        HoverTool(tooltips = [
            ('date', '@prediction_dt{%F}'),
            ('min', f'@min_{metric}'),
            ('avg', f'@score_{metric}'),
            ('max', f'@max_{metric}'),
            ('std', f'@std_{metric}')
        ],
        formatters = {'@prediction_dt': 'datetime'}),
    )
    
    return fig