"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from dash.dependencies import Input, Output, ClientsideFunction, State
import os
from .layout import html_layout


MONGO_HOST = os.environ["MONGO_HOST"]
MONGO_PORT = int(os.environ["MONGO_PORT"])

CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
DB = CLIENT.impact
DEFAULT_INPUT="distgen:n_particle"
DEFAULT_OUTPUT="end_sigma_x"
EXCLUDE_INPUTS = ["mpi_run"]
EXCLUDE_OUTPUTS = ["plot_file", "fingerprint", "archive"]

DF = pd.DataFrame()
ALL_INPUTS = [DEFAULT_INPUT]
ALL_OUTPUTS = [DEFAULT_OUTPUT]

CARD_INDEX = 0


def build_card():
    global CARD_INDEX
    print(CARD_INDEX)

    input_options=[{'label': input_item, 'value': input_item} for input_item in ALL_INPUTS]
    output_options=[{'label': output_item, 'value': output_item} for output_item in ALL_OUTPUTS]

    card = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Title", id=f"card-{CARD_INDEX}"),
                    html.Div(
                        dcc.Dropdown(
                            id=f'input-{CARD_INDEX}',
                            options=input_options,
                            value=DEFAULT_INPUT,
                            multi=True
                        ), 
                        style={'width': '49%', 'display': 'inline-block'}
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id=f'output-{CARD_INDEX}',
                            options=output_options,
                            value=DEFAULT_OUTPUT,
                            multi=True
                            )
                        , style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
                    ),
                ]
            )
        )
    CARD_INDEX += 1
    return card


def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [ (k, v) for k, v in flatten_dict(value).items() ]
        else:
            return [ (key, value) ]

    items = [ item for k, v in d.items() for item in expand(k, v) ]

    return dict(items)


def build_df():
    #get data
    global DF, ALL_INPUTS, ALL_OUTPUTS
    results = DB.results
    results = list(results.find())

    flattened = [flatten_dict(res) for res in results]

    DF = pd.DataFrame(flattened)

    # Load DataFrame
    DF["date"] = pd.to_datetime(DF["isotime"])
    DF = DF.sort_values(by="date")

    ALL_INPUTS = []
    ALL_OUTPUTS = []
    for res in results:
        ALL_INPUTS += list(res["inputs"].keys())
        ALL_OUTPUTS += list(res["outputs"].keys())
    ALL_INPUTS=set(ALL_INPUTS)
    ALL_OUTPUTS=set(ALL_OUTPUTS)

    # drop all unused outputs
    for rem_output in EXCLUDE_OUTPUTS: ALL_OUTPUTS.remove(rem_output)
    for rem_input in EXCLUDE_INPUTS: ALL_INPUTS.remove(rem_input)


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    build_df()

    # Custom HTML layout
    app.index_string = html_layout

    input_options=[{'label': input_item, 'value': input_item} for input_item in ALL_INPUTS]
    output_options=[{'label': output_item, 'value': output_item} for output_item in ALL_OUTPUTS]

    # Create Layout
    app.layout = html.Div(
        children=[
            html.Div(
                html.Img(
                    id="dash-image",
                    src= DF["plot_file"].iloc[0],
                    style={'width': '75%', 'display': 'inline-block'}
                ),
                style={'textAlign': 'center'}
            ),
            html.Div(
            dcc.Dropdown(
                id='input',
                options=input_options,
                value=DEFAULT_INPUT,
                multi=True
            ), 
            style={'width': '49%', 'display': 'inline-block'}
            ),
            html.Div(
            dcc.Dropdown(
                id='output',
                options=output_options,
                value=DEFAULT_OUTPUT,
                multi=True
                )
            , style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(
                    id="time-series",
                    figure=get_time_series(DEFAULT_OUTPUT, None)
                ),
                style={'width': '49%', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(
                    id="scatter-plot",
                    figure=get_scatter(DEFAULT_INPUT, DEFAULT_OUTPUT, None)
                ),
                style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
            ),
        
            html.Button(
                "+", id='submit-val', n_clicks=0
            ),
            html.Div(
                id = 'card-container',
                children = []
  #          create_data_table(DF),
            )
        ],
    )

    init_callbacks(app)

    return app.server



def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id="database-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        sort_mode="native",
        page_size=300,
    )
    return table

def init_callbacks(app):

    @app.callback(
        Output(component_id='scatter-plot', component_property='figure'),
        Output(component_id='time-series', component_property='figure'),
        Input(component_id='input', component_property='value'),
        Input(component_id='output', component_property='value')
    )
    def update_plots(input_value, output_value):
        return [get_scatter(input_value, output_value, None), get_time_series(output_value, None)]

    
    @app.callback(
        Output('dash-image', 'src'),
        Output('scatter-plot', 'figure'),
        Output('time-series', 'figure'),
        Input(component_id='input', component_property='value'),
        Input(component_id='output', component_property='value'),
        Input('scatter-plot', 'selectedData'),
        Input('time-series', 'selectedData'),
        )
    def update_selection(input_val, output_val, selectedData1, selectedData2):

        context = dash.callback_context
        triggered = context.triggered[0]

        if triggered['prop_id'] in ['time-series.selectedData', 'scatter-plot.selectedData']:
            if triggered["value"]:
                selected_points = triggered["value"]["points"]
                selected_points = [point["pointIndex"] for point in selected_points]

                return [
                    dash.no_update,
                    get_scatter(input_val, output_val, selected_points),
                    get_time_series(output_val, selected_points)
                ]

        return [dash.no_update, dash.no_update, dash.no_update]

    @app.callback(
            Output('card-container', 'children'),
            [ Input('submit-val', 'n_clicks')],
            [State('card-container', 'children')]
        )
    def add_card(n_clicks, children):
        
        if children is None:
            children = []
        card = build_card()

        return children.append(card)


def get_scatter(x_col, y_col, selectedpoints):
    fig = px.scatter(DF, x=x_col, y=y_col)

    if selectedpoints is not None:
        selected_df = DF.iloc[selectedpoints]
    else:
        selected_df = DF
        selectedpoints=[]

    selection_bounds = {'x0': np.min(selected_df[x_col]), 'x1': np.max(selected_df[x_col]),
                        'y0': np.min(selected_df[y_col]), 'y1': np.max(selected_df[y_col])}

    fig.update_traces(selectedpoints=selectedpoints,
                      mode='markers+text', 
                      marker={ 'color': 'rgba(214, 116, 0, 0.7)', 'size': 20 }, 
                      unselected={'marker': { 'color': 'rgba(0, 116, 217, 0.3)'}, 
                      'textfont': { 'color': 'rgba(0, 0, 0, 0)' }}
                    )

    fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                       **selection_bounds
                       ))

    return fig

    
def get_time_series(y_col, selectedpoints):
    fig = px.scatter(DF, x="date", y=y_col)


    if selectedpoints is not None:
        selected_df = DF.iloc[selectedpoints]

    else:
        selected_df = DF
        selectedpoints = []

    selection_bounds = {'x0': np.min(selected_df["date"]), 'x1': np.max(selected_df["date"]),
                            'y0': np.min(selected_df[y_col]), 'y1': np.max(selected_df[y_col])}




    fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                       **selection_bounds
                       ))
    
    fig.update_traces(selectedpoints=selectedpoints,
                      mode='markers+text', 
                      marker={ 'color': 'rgba(214, 116, 0, 0.7)', 'size': 20 }, 
                      unselected={'marker': { 'color': 'rgba(0, 116, 217, 0.3)'}, 
                      'textfont': { 'color': 'rgba(0, 0, 0, 0)' }}
                    )

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)