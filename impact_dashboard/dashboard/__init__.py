"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import numpy as np
import json
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from dash.dependencies import Input, Output, ClientsideFunction, State, MATCH, ALL
import os
from .layout import html_layout


MONGO_HOST = os.environ["MONGO_HOST"]
MONGO_PORT = int(os.environ["MONGO_PORT"])

CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
DB = CLIENT.impact
DEFAULT_INPUT="distgen:n_particle"
DEFAULT_OUTPUT="end_sigma_x"
EXCLUDE_INPUTS = ["mpi_run", "header:Nprow", "header:Npcol", "error", "header:Ny", "header:Nx", "header:Nz", "use_mpi", "change_timestep_1:dt", "timeout", "distgen:xy_dist:file"]
EXCLUDE_OUTPUTS = ["plot_file", "fingerprint", "archive", "isotime"]

DF = pd.DataFrame()
ALL_INPUTS = [DEFAULT_INPUT]
ALL_OUTPUTS = [DEFAULT_OUTPUT]

CARD_INDEX = 0


def build_card(selected_data=None):
    global CARD_INDEX

    input_options=[{'label': input_item, 'value': input_item} for input_item in ALL_INPUTS]
    output_options=[{'label': output_item, 'value': output_item} for output_item in ALL_OUTPUTS]

    card = dbc.Col(dbc.Card(
        dbc.CardBody(
            [
            html.Div(
                dcc.Dropdown(
                    id={
                        'type': 'dynamic-input',
                        'index': CARD_INDEX,
                    },
                    options=input_options,
                    value=DEFAULT_INPUT,
                ), 
                style={'width': '49%', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Dropdown(
                    id={
                        'type': 'dynamic-output',
                        'index': CARD_INDEX,
                    },
                    options=output_options,
                    value=DEFAULT_OUTPUT,
                    ),
                    style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(
                    id={
                        'type': 'scatter-plot',
                        'index': CARD_INDEX,
                    },
                    figure=get_scatter(DEFAULT_INPUT, DEFAULT_OUTPUT, selected_data)
                ),
                style={'width': '100%', 'display': 'inline-block'}
            )
        ]
        )
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
    DF["_id"] = DF["_id"].astype(str)
    DF = DF.sort_values(by="date")

    ALL_INPUTS = ["date"]
    ALL_OUTPUTS = []
    for res in results:
        ALL_INPUTS += list(res["inputs"].keys())
        ALL_OUTPUTS += list(res["outputs"].keys())
    ALL_INPUTS=set(ALL_INPUTS)
    ALL_OUTPUTS=set(ALL_OUTPUTS)

    # drop all unused outputs
    for rem_output in EXCLUDE_OUTPUTS: 
        try: ALL_OUTPUTS.remove(rem_output) 
        except: pass
    for rem_input in EXCLUDE_INPUTS: 
        try: ALL_INPUTS.remove(rem_input) 
        except: pass


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
            dbc.themes.BOOTSTRAP
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
            dbc.Row(
                className="row row-cols-3",
                id="dynamic-plots",
                children=[build_card()]
            ),
            html.Div(
                
                html.Button(
                    "+", id='submit-val', n_clicks=0
                ),
                style={'padding': 10}
            ),
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
        Output({'type': 'scatter-plot', 'index': ALL}, 'figure'),
        Output("dash-image", 'src'),
        Input({'type': 'dynamic-input', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-output', 'index': ALL}, 'value'),
        Input({'type': 'scatter-plot', 'index': ALL}, 'selectedData'),
        Input({'type': 'scatter-plot', 'index': ALL}, 'clickData'),
    )
    def update_plot(input_value, output_value, plot_selected_data, plot_click_data):
        context = dash.callback_context
        triggered = context.triggered[0]
        selected_points = []
        updated = [dash.no_update for i in range(len(plot_selected_data))]

        if ".selectedData" in triggered['prop_id']:
            if triggered["value"]:
                selected_points = triggered["value"]["points"]
                selected_points = [point["pointIndex"] for point in selected_points]

                return [
                    get_scatter(input_value[i], output_value[i], selected_points) for i in range(len(plot_selected_data))
                ], dash.no_update

            else:
                return [
                    get_scatter(input_value[i], output_value[i], None) for i in range(len(plot_selected_data))
                ], dash.no_update


        elif ".value" in triggered['prop_id']:
            prop_id = json.loads(triggered['prop_id'].replace(".value", ""))
            prop_idx = prop_id["index"]
            selected_points = None
            if plot_selected_data[0] is not None and plot_selected_data[0]["points"] is not None:
                selected_points = [point["pointIndex"] for point in plot_selected_data[0]["points"]]
            updated[prop_idx] = get_scatter(input_value[prop_idx], output_value[prop_idx], selected_points)
            return updated, dash.no_update

        elif ".clickData" in triggered['prop_id']:
            if triggered["value"]:
                selected_point = triggered["value"]["points"][0]["pointIndex"]

                plot_returns = [get_scatter(input_value[i], output_value[i], [selected_point]) for i in range(len(plot_selected_data))]
                img_file = DF["plot_file"].iloc[selected_point]
                return plot_returns, img_file
        
        else: pass

        return updated, dash.no_update

    @app.callback(
            Output('dynamic-plots', 'children'),
            Input('submit-val', component_property='n_clicks'),
            State('dynamic-plots', 'children')
        )
    def add_card(n_clicks, children):
        if n_clicks==0:
            raise dash.exceptions.PreventUpdate

        if children is None:
            children = []
        card = build_card()

        return children + [card]


def get_scatter(x_col, y_col, selectedpoints):
    fig = px.scatter(DF, x=x_col, y=y_col, hover_data=["_id"])

    if selectedpoints is not None:
        selected_df = DF.iloc[selectedpoints]
        selection_bounds = {'x0': np.min(selected_df[x_col]), 'x1': np.max(selected_df[x_col]),
                        'y0': np.min(selected_df[y_col]), 'y1': np.max(selected_df[y_col])}
        fig.add_shape(dict({'type': 'rect',
                    'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                **selection_bounds
                ))

    else:
        selectedpoints=[]

    fig.update_traces(selectedpoints=selectedpoints,
                      mode='markers+text', 
                      marker={ 'color': 'rgba(214, 116, 0, 0.7)', 'size': 20 }, 
                      unselected={'marker': { 'color': 'rgba(0, 116, 217, 0.3)'}, 
                      'textfont': { 'color': 'rgba(0, 0, 0, 0)' }}
                    )

    fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode="closest")

    return fig



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


if __name__ == '__main__':
    app.run_server(debug=True)