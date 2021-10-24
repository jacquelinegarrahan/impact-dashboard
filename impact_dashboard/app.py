"""Instantiate a Dash app."""
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
from dash import dcc
from dash.dash_table.Format import Format, Scheme
from flask_caching import Cache
import numpy as np
import json
import math
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from dash.dependencies import Input, Output, ClientsideFunction, State, MATCH, ALL
import os
import copy
from impact_dashboard.layout import html_layout
from impact_dashboard import CONFIG
import dash_defer_js_import as dji

from pmd_beamphysics.labels import texlabel

MONGO_HOST = os.environ["MONGO_HOST"]
MONGO_PORT = int(os.environ["MONGO_PORT"])


DEFAULT_INPUT = "distgen:n_particle"
DEFAULT_OUTPUT = "end_sigma_x"

# To exclude from all outputs
EXCLUDE_ALL_OUTPUTS = ["plot_file", "fingerprint", "isotime"]
EXCLUDE_PLOT_INPUTS = []
EXCLUDE_PLOT_OUTPUTS = ["plot_file", "fingerprint", "archive", "isotime"]

latex_refresh_script = dji.Import(src="./assets/mathjax_test.js")
mathjax_script = dji.Import(
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG"
)

app = dash.Dash(
    external_stylesheets=[dbc.themes.DARKLY],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG",
    ],
)

cache = Cache(
    app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)

TIMEOUT = 10

# required for building df
def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [(k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
DB = CLIENT.impact

# build outputs/inputs
results = DB.results
results = list(results.find())

ALL_INPUTS = [
    "date",
    "_id",
    "L0A_phase:dtheta0_deg",
    "L0A_scale:voltage",
    "L0B_phase:dtheta0_deg",
    "L0B_scale:voltage",
    "QA01:b1_gradient",
    "QA02:b1_gradient",
    "QE01:b1_gradient",
    "QE02:b1_gradient",
    "QE03:b1_gradient",
    "QE04:b1_gradient",
    "CQ01:b1_gradient",
    "SQ01:b1_gradient",
    "SOL1:solenoid_field_scale",
    "distgen:t_dist:length:value",
    "distgen:n_particle",
    "distgen:xy_dist:file",
    "change_timestep_1:dt",
]

ALL_OUTPUTS = [
    "end_max_r",
    "end_sigma_z",
    "end_max_amplitude_x",
    "end_mean_beta",
    "plot_file",
    "end_norm_emit_z",
    "end_max_amplitude_z",
    "end_loadbalance_min_n_particle",
    "end_norm_emit_y",
    "end_sigma_px",
    "end_cov_x__px",
    "end_sigma_x",
    "end_sigma_y",
    "end_mean_px",
    "end_moment3_y",
    "end_moment4_z",
    "end_max_amplitude_py",
    "end_mean_py",
    "run_time",
    "end_norm_emit_4d",
    "end_moment3_x",
    "end_mean_pz",
    "end_mean_z",
    "end_loadbalance_max_n_particle",
    "end_moment4_py",
    "end_moment3_py",
    "end_moment3_z",
    "end_total_charge",
    "fingerprint",
    "archive",
    "end_max_amplitude_pz",
    "end_sigma_py",
    "end_mean_y",
    "error",
    "end_moment3_px",
    "end_mean_gamma",
    "end_cov_z__pz",
    "end_moment4_px",
    "end_higher_order_energy_spread",
    "isotime",
    "end_mean_x",
    "end_sigma_pz",
    "end_t",
    "end_n_particle_loss",
    "end_norm_emit_xy",
    "end_norm_emit_x",
    "end_mean_kinetic_energy",
    "end_n_particle",
    "end_moment4_y",
    "end_sigma_gamma",
    "end_moment3_pz",
    "end_cov_y__py",
    "end_moment4_pz",
    "end_max_amplitude_px",
    "end_moment4_x",
    "end_max_amplitude_y",
]


DROPDOWN_INPUTS = copy.copy(ALL_INPUTS)
DROPDOWN_OUTPUTS = copy.copy(ALL_OUTPUTS)


# drop all exclusions
for rem_output in EXCLUDE_ALL_OUTPUTS:
    try:
        DROPDOWN_OUTPUTS.remove(rem_output)
    except:
        pass

# drop all plot exclusions
for rem_output in EXCLUDE_PLOT_OUTPUTS:
    try:
        DROPDOWN_OUTPUTS.remove(rem_output)
    except:
        pass
for rem_input in EXCLUDE_PLOT_INPUTS:
    try:
        DROPDOWN_INPUTS.remove(rem_input)
    except:
        pass

CARD_COUNT = 0
CARD_INDICES = {}
LIVE_CARD_COUNT = 0

TABLE_DEFAULTS = [
    "date",
    "SOL1:solenoid_field_scale",
    "SQ01:b1_gradient",
    "CQ01:b1_gradient",
    "QA01:b1_gradient",
    "QA02:b1_gradient",
    "QE01:b1_gradient",
    "QE02:b1_gradient",
    "QE03:b1_gradient",
    "QE04:b1_gradient",
    "end_norm_emit_x",
    "end_norm_emit_y",
    "end_norm_emit_z",
]


def get_label(item: str):
    """Utility function for stripping end prefix from label and formatting
    for LaTeX rendering.
    """

    if "end_" in item:
        item = item.replace("end_", "")

    label = texlabel(item)
    if not label:
        return item
    
    else:
        return f"$${label}$$"


def build_card(df, x: str = None, y: str = None, selected_data: list = None):
    """ Representations of plot cards used for displaying data

    Args:
        x (str): Variable for plot's x axis
        y (str): Variable for plot's y axis
        selected_data (list): Selected points for sharing across plots

    """

    # Global variable card index tracks the number of cards created
    global CARD_COUNT

    options = [
        {"label": get_label(item), "value": item} for item in ALL_INPUTS + ALL_OUTPUTS
    ]

    # use default input/output when creating a card
    if not x:
        x = DEFAULT_INPUT
    if not y:
        y = DEFAULT_OUTPUT

    card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.Label(
                                    ["X:"],
                                    style={
                                        "display": "block",
                                        "textAlign": "center",
                                        "fontSize": int(
                                            CONFIG["card"]["header-font-size"]
                                        ),
                                    },
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id={
                                            "type": "dynamic-x",
                                            "index": CARD_COUNT,
                                        },
                                        options=options,
                                        value=x,
                                        clearable=False,
                                    ),
                                    style={
                                        "width": "100%",
                                        "fontSize": int(CONFIG["card"]["font-size"]),
                                        "margins": 0,
                                    },
                                ),
                            ],
                            style={"marginRight": "2px", "marginLeft": "2px"},
                        ),
                        dbc.Col(
                            children=[
                                html.Label(
                                    ["Y:"],
                                    style={
                                        "display": "block",
                                        "textAlign": "center",
                                        "fontSize": int(
                                            CONFIG["card"]["header-font-size"]
                                        ),
                                    },
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id={
                                            "type": "dynamic-y",
                                            "index": CARD_COUNT,
                                        },
                                        options=options,
                                        value=y,
                                        clearable=False,
                                    ),
                                    style={
                                        "width": "100%",
                                        "fontSize": int(CONFIG["card"]["font-size"]),
                                    },
                                ),
                            ],
                            style={"marginRight": "2px", "marginLeft": "2px"},
                        ),
                        dbc.Col(
                            children=[
                                html.Label(
                                    ["Color by:"],
                                    style={
                                        "display": "block",
                                        "textAlign": "center",
                                        "fontSize": int(
                                            CONFIG["card"]["header-font-size"]
                                        ),
                                    },
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id={
                                            "type": "dynamic-coloring",
                                            "index": CARD_COUNT,
                                        },
                                        options=options,
                                        clearable=True,
                                    ),
                                    style={
                                        "width": "100%",
                                        "display": "inline-block",
                                        "fontSize": int(CONFIG["card"]["font-size"]),
                                    },
                                ),
                            ],
                            style={"marginRight": "2px", "marginLeft": "2px"},
                        ),
                        html.Button(
                            "x",
                            id={"type": "dynamic-remove", "index": CARD_COUNT,},
                            n_clicks=0,
                            style={"height": 25},
                        ),
                    ],
                    no_gutters=True,
                ),
                html.Div(
                    dcc.Graph(
                        id={"type": "scatter-plot", "index": CARD_COUNT,},
                        figure=get_scatter(df, x, y, selected_data),
                        style={"height": CONFIG["card"]["plot-height"]},
                    ),
                    style={"width": "100%", "display": "inline-block"},
                ),
            ]
        ),
    )
    if CARD_INDICES.values():
        CARD_INDICES[CARD_COUNT] = max(CARD_INDICES.values()) + 1
    else:
        CARD_INDICES[CARD_COUNT] = 0
    CARD_COUNT += 1
    return card


def get_scatter(df, x_col, y_col, selectedpoints, color_by=None):
    if color_by == "NONE":
        color_by = None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        hover_data=["_id"],
        color=color_by,
        color_continuous_scale="viridis",
        labels={x_col: get_label(x_col), y_col: get_label(y_col)},
        template=CONFIG["scatter"]["plotly-theme"],
    )

    if selectedpoints is not None:
        selected_df = df.iloc[selectedpoints]
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
                **selection_bounds,
            )
        )

    else:
        selectedpoints = []

    if not color_by:
        fig.update_traces(
            selectedpoints=selectedpoints,
            mode="markers+text",
            marker={"color": CONFIG["scatter"]["selected-marker-color"], "size": 15},
            unselected={
                "marker": {"color": CONFIG["scatter"]["marker-color"]},
                "textfont": {"color": "rgba(0, 0, 0, 0)"},
            },
        )

    else:
        fig.update_traces(mode="markers+text", marker={"size": 15})
        fig.update_coloraxes(colorbar_title_side="right")

    fig.update_layout(
        font_color="grey",
        font_size=10,
        margin={"l": 20, "r": 0, "b": 15, "t": 5},
        dragmode="select",
        hovermode="closest",
    )

    # fig.update_xaxes(title_text=r"$$x (Å)$$")

    return fig


@cache.memoize(timeout=TIMEOUT)
def get_df():
    """Function used for updating the cached dataframe.
    """
    # get data
    CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
    DB = CLIENT.impact
    results = DB.results
    results = list(results.find())

    flattened = [flatten_dict(res) for res in results]
    df = pd.DataFrame(flattened)

    # Load DataFrame
    df["date"] = pd.to_datetime(df["isotime"])
    df["_id"] = df["_id"].astype(str)
    df = df.sort_values(by="date")

    return df.to_json(date_format="iso", orient="split")


def dataframe():
    return pd.read_json(get_df(), orient="split")


def init_dashboard():
    """Create and format Plotly Dash dashboard."""

    # get initial latest cached df
    df = dataframe()

    # create a representation of all inputs for use with table using appropriate number of sig figs
    input_rep = [
        {
            "inputs": get_label(ALL_INPUTS[i]),
            "value": format(df[DROPDOWN_INPUTS[i]].iloc[0], f".{CONFIG['tables']['n_sig_figs']}g"),
        }
        if isinstance(df[DROPDOWN_INPUTS[i]].iloc[0], float)
        else {
            "inputs": get_label(ALL_INPUTS[i]),
            "value": df[DROPDOWN_INPUTS[i]].iloc[0],
        }
        for i in range(len(DROPDOWN_INPUTS))
    ]

    # create a representation of all inputs for use with table using appropriate number of sig figs
    output_rep = [
        {
            "outputs": get_label(ALL_OUTPUTS[i]),
            "value": format(df[DROPDOWN_OUTPUTS[i]].iloc[0], f".{CONFIG['tables']['n_sig_figs']}g"),
        }
        if isinstance(df[DROPDOWN_OUTPUTS[i]].iloc[0], float)
        else {
            "outputs": get_label(ALL_OUTPUTS[i]),
            "value": df[DROPDOWN_OUTPUTS[i]].iloc[0],
        }
        for i in range(len(DROPDOWN_OUTPUTS))
    ]

    # Format columns for explore table indicating number of sig figs for numeric types
    explore_table_cols = [
        {
            "name": i,
            "id": i,
            "type": "numeric",
            "format": Format(precision=int(CONFIG["tables"]["n_sig_figs"]), scheme=Scheme.decimal),
        }
        if df.dtypes[i] in ["int", "float64"]
        else {"name": i, "id": i}
        for i in df[TABLE_DEFAULTS].columns
    ]

    # Custom HTML layout
    app.index_string = html_layout


    # Create Layout
    app.layout = html.Div(
        [
            dbc.Row(
                children=[
                    html.Img(
                        id="dash-image",
                        src=df["plot_file"].iloc[0],
                        style={"width": CONFIG["dash"]["width"], "height": CONFIG["dash"]["height"]},
                    ),
                    dash_table.DataTable(
                        id="input-table",
                        columns=[
                            {"name": "inputs", "id": "inputs"},
                            {"name": "value", "id": "value"},
                        ],
                        data=input_rep,
                        sort_action="native",
                        style_table={
                            "overflowY": "auto",
                            "overflowX": "auto",
                            "height":  CONFIG["dash"]["height"],
                            "width": CONFIG["tables"]["width"],
                            "display": "inline-block",
                            "maxHeight":  CONFIG["dash"]["height"],
                        },
                        style_header={
                            "backgroundColor": CONFIG["tables"][
                                "header-background-color"
                            ]
                        },
                        style_cell={
                            "backgroundColor": CONFIG["tables"][
                                "cell-background-color"
                            ],
                            "color": CONFIG["tables"]["text-color"],
                            "fontSize": int(CONFIG["tables"]["font-size"]),
                            "textAlign": "left",
                        },
                        fixed_rows={"headers": True},
                    ),
                    dash_table.DataTable(
                        id="output-table",
                        columns=[
                            {"name": "outputs", "id": "outputs"},
                            {"name": "value", "id": "value"},
                        ],
                        data=output_rep,
                        sort_action="native",
                        style_table={
                            "overflowY": "auto",
                            "overflowX": "auto",
                            "width": CONFIG["tables"]["width"],
                            "height":  CONFIG["dash"]["height"],
                            "maxHeight": CONFIG["dash"]["height"],
                        },
                        style_header={
                            "backgroundColor": CONFIG["tables"][
                                "header-background-color"
                            ]
                        },
                        style_cell={
                            "backgroundColor": CONFIG["tables"][
                                "cell-background-color"
                            ],
                            "color": CONFIG["tables"]["text-color"],
                            "fontSize": int(CONFIG["tables"]["font-size"]),
                            "textAlign": "left",
                        },
                        fixed_rows={"headers": True},
                    ),
                ],
                style={"height": CONFIG["dash"]["height"]},
            ),
            dbc.Row(
                className="row row-cols-4",
                id="dynamic-plots",
                children=[
                    build_card(df, x="CQ01:b1_gradient", y="SQ01:b1_gradient"),
                    build_card(df, x="SOL1:solenoid_field_scale", y="end_norm_emit_y"),
                    build_card(df, x="QA01:b1_gradient", y="QA02:b1_gradient"),
                    build_card(df, x="QE01:b1_gradient", y="QE02:b1_gradient"),
                    build_card(df, x="QE03:b1_gradient", y="QE04:b1_gradient"),
                    html.Div(
                        html.Button("+", id="submit-val", n_clicks=0),
                        style={"padding": 10},
                    ),
                ],
            ),
            html.Div(
                dcc.Dropdown(
                    id="explore-dropdown",
                    options=[{"label": get_label(i), "value": i} for i in df.columns]
                    + [{"label": "Reset", "value": "Reset"}],
                    value=TABLE_DEFAULTS,
                    clearable=False,
                    multi=True,
                ),
                style={"width": "100%", "fontSize": 20, "margins": 0,},
            ),
            dash_table.DataTable(
                id="explore-table",
                columns=explore_table_cols,
                data=df[TABLE_DEFAULTS].to_dict("records"),
                sort_action="native",
                style_table={
                    "overflowY": "auto",
                    #    "overflowX": "auto",
                    "width": "100vw",
                    "display": "inline-block",
                    #    "autoWidth": "true",
                },
                style_header={
                    "backgroundColor": CONFIG["tables"]["header-background-color"]
                },
                style_cell={
                    "backgroundColor": CONFIG["tables"]["cell-background-color"],
                    "color": CONFIG["tables"]["text-color"],
                    "fontSize": int(CONFIG["tables"]["font-size"]),
                    "textAlign": "left",
                    "minWidth": "180px",
                    "width": "180px",
                    "maxWidth": "180px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                fixed_rows={"headers": True},
            ),
            ###### important for latex ######
            latex_refresh_script,
            mathjax_script,
        ]
    )                   


# initialize dashboard
init_dashboard()


# REGISTER APP CALLBACKS

@app.callback(
    Output({"type": "scatter-plot", "index": ALL}, "figure"),
    Output("dash-image", "src"),
    Output("input-table", "data"),
    Output("output-table", "data"),
    Input({"type": "dynamic-x", "index": ALL}, "value"),
    Input({"type": "dynamic-y", "index": ALL}, "value"),
    Input({"type": "scatter-plot", "index": ALL}, "selectedData"),
    Input({"type": "scatter-plot", "index": ALL}, "clickData"),
    Input({"type": "dynamic-coloring", "index": ALL}, "value"),
)
def update_data(
    input_value, output_value, plot_selected_data, plot_click_data, color_by
):
    """Handles data updates and synchronizes across elements.

    """
    # use triggered context to get the origin of the action
    context = dash.callback_context
    triggered = context.triggered[0]

    selected_points = []

    # list of no updates markers to pass to the x values for the card plots
    updated = [dash.no_update for i in range(len(plot_selected_data))]

    # get latest cached dataframe
    df = dataframe()

    # data has been selected, update selection across cards
    if ".selectedData" in triggered["prop_id"]:
        if triggered["value"]:
            selected_points = triggered["value"]["points"]
            selected_points = [point["pointIndex"] for point in selected_points]

            return (
                [
                    get_scatter(
                        df, input_value[i], output_value[i], selected_points, None
                    )
                    for i in range(len(plot_selected_data))
                ],
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        else:
            return (
                [
                    get_scatter(df, input_value[i], output_value[i], None, color_by[i])
                    for i in range(len(plot_selected_data))
                ],
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    # A value has been updated in a dropdown, update the single scatter plot
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

        # update the scatter plot at a given index, all others remain same
        updated[CARD_INDICES[prop_idx]] = get_scatter(
            df,
            input_value[CARD_INDICES[prop_idx]],
            output_value[CARD_INDICES[prop_idx]],
            selected_points,
            color_by[CARD_INDICES[prop_idx]],
        )
        return updated, dash.no_update, dash.no_update, dash.no_update

    # a single point has been clicked, update selection across cards, 
    # associated image with point, and input/output data tables
    elif ".clickData" in triggered["prop_id"]:
        if triggered["value"]:
            selected_point = triggered["value"]["points"][0]["pointIndex"]

            plot_returns = [
                get_scatter(df, input_value[i], output_value[i], [selected_point], None)
                for i in range(len(plot_selected_data))
            ]

            img_file = df["plot_file"].iloc[selected_point]

            # update data tables
            input_rep = [
                {
                    "inputs": get_label(ALL_INPUTS[i]),
                    "value": df[ALL_INPUTS[i]].iloc[selected_point],
                }
                for i in range(len(ALL_INPUTS))
            ]
            output_rep = [
                {
                    "outputs": get_label(ALL_OUTPUTS[i]),
                    "value": df[ALL_OUTPUTS[i]].iloc[selected_point],
                }
                for i in range(len(ALL_OUTPUTS))
            ]

            return plot_returns, img_file, input_rep, output_rep

    else:
        pass

    return updated, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("dynamic-plots", "children"),
    Input("submit-val", component_property="n_clicks"),
    Input({"type": "dynamic-remove", "index": ALL}, component_property="n_clicks"),
    State({"type": "dynamic-remove", "index": ALL}, "id"),
    State("dynamic-plots", "children"),
)
def update_cards(n_clicks, n_clicks_remove, remove_id, children):
    """Updates number of cards if addition/deletion

    """
    # prevent update if no clicks
    if n_clicks == 0 and not any(n_clicks_remove):
        raise dash.exceptions.PreventUpdate

    if children is None:
        children = []

    df = dataframe()

    # use context to see if remove button was pushed
    context = dash.callback_context
    triggered = context.triggered[0]

    # if removed must adjust all card indices \
    if "dynamic-remove" in triggered["prop_id"]:
        prop_id = json.loads(triggered["prop_id"].replace(".n_clicks", ""))
        prop_idx = prop_id["index"]
        card_idx = CARD_INDICES.pop(prop_idx)
        children.pop(card_idx)

        # update card indices for all greater values
        for item in CARD_INDICES.keys():
            if CARD_INDICES[item] > card_idx:
                CARD_INDICES[item] -= 1

    # otherwise, a card has been added
    else:
        card = build_card(df)
        children.insert(-1, card)

    return children


@app.callback(
    Output("explore-table", "data"),
    Output("explore-table", "columns"),
    Output("explore-dropdown", "value"),
    Input("explore-dropdown", "value"),
)
def update_explore_table(selected_values):
    """Handles updates to the exploration table dropdown and updates table accordingly.

    """

    # get latest cached dataframe
    df = dataframe()

    # Handle table reset to defaults
    if "Reset" in selected_values:
        df = df[TABLE_DEFAULTS]
        selection = TABLE_DEFAULTS
    else:
        df = df[selected_values]
        selection = selected_values

    # format columns
    columns = [
        {
            "name": i,
            "id": i,
            "type": "numeric",
            "format": Format(precision=int(CONFIG["tables"]["n_sig_figs"]), scheme=Scheme.decimal),
        }
        if df.dtypes[i] in ["int", "float64"]
        else {"name": i, "id": i}
        for i in df.columns
    ]
    data = df.to_dict("records")

    return data, columns, selection

def main():
    """Main method for app entrypoint
    """
    app.run_server(host="0.0.0.0")

if __name__ == "__main__":
    main()
