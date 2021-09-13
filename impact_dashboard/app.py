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
from impact_dashboard.layout import html_layout


MONGO_HOST = os.environ["MONGO_HOST"]
MONGO_PORT = int(os.environ["MONGO_PORT"])

CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
DB = CLIENT.impact
DEFAULT_INPUT = "distgen:n_particle"
DEFAULT_OUTPUT = "end_sigma_x"
EXCLUDE_INPUTS = [
    "mpi_run",
    "header:Nprow",
    "header:Npcol",
    "error",
    "header:Ny",
    "header:Nx",
    "header:Nz",
    "use_mpi",
    "change_timestep_1:dt",
    "timeout",
    "distgen:xy_dist:file",
]
EXCLUDE_OUTPUTS = ["plot_file", "fingerprint", "archive", "isotime"]


DF = pd.DataFrame()
ALL_INPUTS = [DEFAULT_INPUT]
ALL_OUTPUTS = [DEFAULT_OUTPUT]

CARD_COUNT=0
CARD_INDICES ={}


LABELS = {
    #   '_id',
    #   'isotime',
    #   'distgen:t_dist:length:value',
    #   'distgen:n_particle',
    #   'stop',
    #   'timeout',
    #   'header:Nx',
    #   'header:Ny',
    #   'header:Nz',
    #   'change_timestep_1:dt',
    #   'header:Nprow',
    #   'header:Npcol',
    #   'use_mpi',
    #   'mpi_run',
    #   'SOL1:solenoid_field_scale',
    #   'CQ01:b1_gradient',
    #   'SQ01:b1_gradient',
    #   'L0A_phase:dtheta0_deg',
    #   'L0B_phase:dtheta0_deg',
    #   'L0A_scale:voltage',
    #   'L0B_scale:voltage',
    #   'QA01:b1_gradient',
    #   'QA02:b1_gradient',
    #   'QE01:b1_gradient',
    #   'QE02:b1_gradient',
    #   'QE03:b1_gradient',
    #    'QE04:b1_gradient',
    #   'distgen:xy_dist:file',
    #   'impact_config',
    #   'distgen_input_file',
    #   'SOLN:IN20:121:BDES',
    #   'QUAD:IN20:121:BDES',
    #   'QUAD:IN20:122:BDES',
    #   'ACCL:IN20:300:L0A_PDES',
    #   'ACCL:IN20:400:L0B_PDES',
    #   'ACCL:IN20:300:L0A_ADES',
    #   'ACCL:IN20:400:L0B_ADES',
    #   'QUAD:IN20:361:BDES',
    #   'QUAD:IN20:371:BDES',
    #   'QUAD:IN20:425:BDES',
    #   'QUAD:IN20:441:BDES',
    #   'QUAD:IN20:511:BDES',
    #   'QUAD:IN20:525:BDES',
    #   'error',
    #   'end_t',
    #   'end_mean_z',
    #   'end_sigma_z',
    #   'end_norm_emit_z',
    #   'end_loadbalance_min_n_particle',
    #   'end_loadbalance_max_n_particle',
    #   'end_n_particle',
    #   'end_moment3_x',
    #   'end_moment3_y',
    #   'end_moment3_z',
    #   'end_max_amplitude_x',
    #   'end_max_amplitude_y',
    #   'end_max_amplitude_z',
    #   'end_mean_x',
    #   'end_sigma_x',
    #   'end_norm_emit_x',
    #   'end_mean_y',
    #   'end_sigma_y',
    #   'end_norm_emit_y',
    #   'end_mean_gamma',
    #   'end_mean_beta',
    #   'end_max_r',
    #   'end_sigma_gamma',
    #   'end_moment4_x',
    #   'end_moment4_y',
    #   'end_moment4_z',
    #   'end_mean_pz',
    #   'end_sigma_pz',
    #   'end_cov_z__pz',
    #   'end_moment3_px',
    #   'end_moment3_py',
    #   'end_moment3_pz',
    "end_max_amplitude_px": "$end max amplitude p_x$",
    "end_max_amplitude_py": "$end max amplitude p_y$",
    "end_max_amplitude_pz": "$end max amplitude p_z$",
    #   'end_mean_px',
    #   'end_sigma_px',
    #   'end_cov_x__px',
    #   'end_mean_py',
    #   'end_sigma_py',
    #   'end_cov_y__py',
    #   'end_mean_kinetic_energy',
    #   'end_moment4_px',
    #   'end_moment4_py',
    #   'end_moment4_pz',
    #   'run_time',
    #   'end_n_particle_loss',
    #   'end_total_charge',
    #   'end_higher_order_energy_spread',
    #   'end_norm_emit_xy',
    #   'end_norm_emit_4d',
}


def build_card(x: str = None, y: str = None, selected_data: list = None):
    """ Representations of plot cards used for displaying data

    Args:
        x (str): Variable for plot's x axis
        y (str): Variable for plot's y axis
        selected_data (list): Selected points for sharing across plots

    """

    # Global variable card index tracks the number of cards created
    global CARD_COUNT

    options = [
        {"label": input_item, "value": input_item}
        for input_item in set.union(ALL_INPUTS, ALL_OUTPUTS)
    ]

    # use default input/output when creating a card
    if not x:
        x = DEFAULT_INPUT
    if not y:
        y = DEFAULT_OUTPUT

    card = dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        html.Button("x", id={"type": "dynamic-remove", "index": CARD_COUNT,}, n_clicks=0, style={"height": 25}),
                        style={"padding": 0},
                        justify="end",
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id={"type": "dynamic-input", "index": CARD_COUNT,},
                            options=options,
                            value=x,
                            clearable=False,
                        ),
                        style={
                            "width": "32%",
                            "display": "inline-block",
                            "font-size": "10px",
                        },
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id={"type": "dynamic-output", "index": CARD_COUNT,},
                            options=options,
                            value=y,
                            clearable=False,
                        ),
                        style={
                            "width": "32%",
                            "display": "inline-block",
                            "font-size": "10px",
                        },
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id={"type": "dynamic-coloring", "index": CARD_COUNT,},
                            options=options + [{"label": "None", "value": "None"}],
                            value= "None",
                            clearable=False,
                        ),
                        style={
                            "width": "32%",
                            "display": "inline-block",
                            "font-size": "10px",
                        },
                    ),
                    html.Div(
                        dcc.Graph(
                            id={"type": "scatter-plot", "index": CARD_COUNT,},
                            figure=get_scatter(x, y, selected_data),
                            style={"height": "40vh"},

                        ),
                        style={"width": "100%", "display": "inline-block"},
                    ),
                ]
            )
        ),
    )
    CARD_INDICES[CARD_COUNT] = CARD_COUNT
    CARD_COUNT += 1
    return card


def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [(k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def build_df():
    # get data
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
    ALL_INPUTS = set(ALL_INPUTS)
    ALL_OUTPUTS = set(ALL_OUTPUTS)

    # drop all unused outputs
    for rem_output in EXCLUDE_OUTPUTS:
        try:
            ALL_OUTPUTS.remove(rem_output)
        except:
            pass
    for rem_input in EXCLUDE_INPUTS:
        try:
            ALL_INPUTS.remove(rem_input)
        except:
            pass


def init_dashboard():
    """Create a Plotly Dash dashboard."""
    # pass our own flask server instead of using Dash's
    app = dash.Dash(
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
            dbc.themes.BOOTSTRAP,
        ],
        external_scripts=[
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
        ],
    )

    build_df()

    input_res = list(DB.results.find())[0]["inputs"]
    inputs = list(input_res.keys())
    input_values = list(input_res.values())

    input_rep = [{"inputs": inputs[i], "value": input_values[i]} for i in range(len(inputs))]

    output_res = list(DB.results.find())[0]["outputs"]
    outputs = list(output_res.keys())
    output_values = list(output_res.values())
    output_rep = [{"outputs": outputs[i], "value": output_values[i]} for i in range(len(outputs))]

    # Custom HTML layout
    app.index_string = html_layout

    # Create Layout
    app.layout = html.Div(
        children=[
            dbc.Row(
                children= [
                    html.Img(
                        id="dash-image",
                        src=DF["plot_file"].iloc[0],
                        style={"width": "50%", "display": "inline-block"},
                    ),
                    dash_table.DataTable(
                        id="input-table",
                        columns=[{"name": "inputs", "id": "inputs"}, {"name": "value", "id": "value"}],
                        data=input_rep,
                        sort_action="native",
                        page_size=300,
                        style_table={'width': '25%', 'overflowY': 'auto'},
                        fixed_rows={'headers': True},
                    ),
                    dash_table.DataTable(
                        id="output-table",
                        columns=[{"name": "outputs", "id": "outputs"}, {"name": "value", "id": "value"}],
                        data=output_rep,
                        sort_action="native",
                        page_size=300,
                        style_table={'width': '25%', 'overflowY': 'auto'},
                        fixed_rows={'headers': True},
                    ),
                ]
            ),
            dbc.Row(
                className="row row-cols-4",
                id="dynamic-plots",
                children=[
                    build_card(x="CQ01:b1_gradient", y="SQ01:b1_gradient"),
                    build_card(x="SOL1:solenoid_field_scale", y="end_norm_emit_y"),
                    build_card(x="QA01:b1_gradient", y="QA02:b1_gradient"),
                    build_card(x="QE01:b1_gradient", y="QE02:b1_gradient"),
                    build_card(x="QE03:b1_gradient", y="QE04:b1_gradient"),
                    html.Div(
                        
                        html.Button(
                            "+", id='submit-val', n_clicks=0
                        ),
                        style={'padding': 10}
                    ),
                ]
            )
        ],
    )

    init_callbacks(app)

    return app


def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): Dataframe for generating table
    
    """
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
        Output({"type": "scatter-plot", "index": ALL}, "figure"),
        Output("dash-image", "src"),
        Input({"type": "dynamic-input", "index": ALL}, "value"),
        Input({"type": "dynamic-output", "index": ALL}, "value"),
        Input({"type": "scatter-plot", "index": ALL}, "selectedData"),
        Input({"type": "scatter-plot", "index": ALL}, "clickData"),
        Input({"type": "dynamic-coloring", "index": ALL}, "value"),
    )
    def update_plot(
        input_value, output_value, plot_selected_data, plot_click_data, color_by
    ):
        context = dash.callback_context

        triggered = context.triggered[0]
        selected_points = []
        updated = [dash.no_update for i in range(len(plot_selected_data))]

        # update selected data
        if ".selectedData" in triggered["prop_id"]:
            if triggered["value"]:
                selected_points = triggered["value"]["points"]
                selected_points = [point["pointIndex"] for point in selected_points]

                return (
                    [
                        get_scatter(
                            input_value[i], output_value[i], selected_points, None
                        )
                        for i in range(len(plot_selected_data))
                    ],
                    dash.no_update,
                )

            else:
                return (
                    [
                        get_scatter(input_value[i], output_value[i], None, color_by[i])
                        for i in range(len(plot_selected_data))
                    ],
                    dash.no_update,
                )

        # update input/output values
        elif ".value" in triggered["prop_id"]:
            prop_id = json.loads(triggered["prop_id"].replace(".value", ""))
            prop_idx = prop_id["index"]

            selected_points = None
            if (
                plot_selected_data[0] is not None
                and plot_selected_data[0]["points"] is not None
            ):
                selected_points = [
                    point["pointIndex"] for point in plot_selected_data[0]["points"]
                ]

            updated[CARD_INDICES[prop_idx]] = get_scatter(
                input_value[CARD_INDICES[prop_idx]],
                output_value[CARD_INDICES[prop_idx]],
                selected_points,
                color_by[CARD_INDICES[prop_idx]],
            )
            return updated, dash.no_update

        elif ".clickData" in triggered["prop_id"]:
            if triggered["value"]:
                selected_point = triggered["value"]["points"][0]["pointIndex"]

                plot_returns = [
                    get_scatter(input_value[i], output_value[i], [selected_point], None)
                    for i in range(len(plot_selected_data))
                ]
                img_file = DF["plot_file"].iloc[selected_point]
                return plot_returns, img_file

        else:
            pass

        return updated, dash.no_update

    @app.callback(
        Output("dynamic-plots", "children"),
        Input("submit-val", component_property="n_clicks"),
        Input({"type": "dynamic-remove", "index": ALL}, component_property="n_clicks"),
        State({'type': 'dynamic-remove', 'index': ALL}, 'id'),
        State("dynamic-plots", "children"),
    )
    def update_cards(n_clicks, n_clicks_remove, remove_id, children):
        # prevent update if no clicks
        if n_clicks == 0 and not any(n_clicks_remove):
            raise dash.exceptions.PreventUpdate

        if children is None:
            children = []

        # use context to see if remove was pushed
        context = dash.callback_context
        triggered = context.triggered[0]

        if "dynamic-remove" in triggered["prop_id"]:
            prop_id = json.loads(triggered["prop_id"].replace(".n_clicks", ""))
            prop_idx = prop_id["index"]
            children.pop(CARD_INDICES[prop_idx])

            # update card indices for all greater values
            for item in CARD_INDICES.keys():
                if CARD_INDICES[item] > CARD_INDICES[prop_idx]:
                    CARD_INDICES[item] -= 1

        else:
            card = build_card()
            children.insert(-1, card)

        return children



def get_scatter(x_col, y_col, selectedpoints, color_by=None):
    if color_by == "NONE":
        color_by = None

    fig = px.scatter(
        DF,
        x=x_col,
        y=y_col,
        hover_data=["_id"],
        color=color_by,
        color_continuous_scale="viridis",
        labels={x_col: LABELS.get(x_col, x_col), y_col: LABELS.get(y_col, y_col)},
    )

    if selectedpoints is not None:
        selected_df = DF.iloc[selectedpoints]
        selection_bounds = {
            "x0": np.min(selected_df[x_col]),
            "x1": np.max(selected_df[x_col]),
            "y0": np.min(selected_df[y_col]),
            "y1": np.max(selected_df[y_col]),
        }
        fig.add_shape(
            dict(
                {
                    "type": "rect",
                    "line": {"width": 1, "dash": "dot", "color": "darkgrey"},
                },
                **selection_bounds
            )
        )

    else:
        selectedpoints = []

    if color_by == "None":
        fig.update_traces(
            selectedpoints=selectedpoints,
            mode="markers+text",
            marker={"color": "rgba(214, 116, 0, 0.7)", "size": 15},
            unselected={
                "marker": {"color": "rgba(0, 116, 217, 0.3)"},
                "textfont": {"color": "rgba(0, 0, 0, 0)"},
            },
        )

    else:
        fig.update_traces(mode="markers+text", marker={"size": 15})
        fig.update_coloraxes(colorbar_title_side="right")

    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        font_size=10,
        margin={"l": 20, "r": 0, "b": 15, "t": 5},
        dragmode="select",
        hovermode="closest"
    )

    return fig


if __name__ == "__main__":
    app = init_dashboard()
    app.run_server(debug=True)
