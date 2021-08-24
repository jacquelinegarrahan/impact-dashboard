"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from dash.dependencies import Input, Output
import os
from .layout import html_layout


MONGO_HOST = os.environ["MONGO_HOST"]
MONGO_PORT = int(os.environ["MONGO_PORT"])

CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
DB = CLIENT.impact
RESULTS = DB.results
RESULTS = list(RESULTS.find())
DEFAULT_PV="distgen:n_particle"

DATA = [res["inputs"][DEFAULT_PV] for res in RESULTS]
ISOTIME = [res["isotime"] for res in RESULTS]

ALL_PVS = []
for res in RESULTS:
    ALL_PVS += res["inputs"].keys()
ALL_PVS=set(ALL_PVS)

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

    #get data
    results = DB.results
    results = list(results.find())

    # Load DataFrame
    df = pd.DataFrame({DEFAULT_PV: DATA, "date": ISOTIME})
    df["date"] = pd.to_datetime(df["date"])
    data = [res["inputs"][DEFAULT_PV] for res in results]
    isotime = [res["isotime"] for res in results]

    all_pvs = []
    for res in results:
        all_pvs += res["inputs"].keys()
    all_pvs=set(all_pvs)


    # Custom HTML layout
    app.index_string = html_layout

    options=[{'label': pv, 'value': pv} for pv in all_pvs]

    fig = px.scatter(df, x="date", y=DEFAULT_PV)

    # Create Layout
    app.layout = html.Div(
        children=[
            dcc.Dropdown(
                id='pv-input',
                options=options,
                value=DEFAULT_PV
            ),
            dcc.Graph(
                id="time-series",
                figure=fig
            ),
            create_data_table(df),
        ],
        id="dash-container",
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
        Output(component_id='time-series', component_property='figure'),
        Input(component_id='pv-input', component_property='value')
    )
    def update_time_series(input_value):
        RESULTS = DB.results
        RESULTS = list(RESULTS.find())

        # Load DataFrame
        DATA = [res["inputs"][input_value] for res in RESULTS]
        df = pd.DataFrame({input_value: DATA, "date": ISOTIME})
        df["date"] = pd.to_datetime(df["date"])

        return px.scatter(df, x="date", y=input_value)