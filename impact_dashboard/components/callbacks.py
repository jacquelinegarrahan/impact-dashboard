


Output({"type": "scatter-plot", "index": ALL}, "figure"),

inputs = {
    "input1": {
        "dash_repr": Input({"type": "scatter-plot", "index": ALL}, "value"),
        "outputs": ["output1", "explore-table"]
    }
    "output1": {
        "dash_repr": Output({"type": "scatter-plot", "index": ALL}, "figure"),
        "callable": get_scatter
        "callable_args": ["dynamic-x", "dynamic-y", "selected_points", "color-by"]
    }
    "explore-table": {
        "dash_repr": 
    }
}
# requires hash map of input/ output to input/output arg index



def register_data_callbacks(app, inputs, outputs):
    @app.callaback(
        *outputs,
        *inputs
    )
    # need to accept variable number of inputs/outputs, should be dict
    def update_data_selection(*input_str_repr, *output_str_repr):
        df = dataframe()

        # use triggered context to get the origin of the action
        context = dash.callback_context
        triggered = context.triggered[0]

        # get id of triggering property
        prop_id = triggered["prop_id"].split(".")[0]

        #create represetation
        output_repr = ...

        # get representation index of prop-id
        prop_idx = ...


        # data has been selected from a plot, update all target selections
        if ".selectedData" in triggered["prop_id"]:
            targets = inputs[prop_id]["outputs"]

            # if has been triggered by value selection
            if triggered["value"]:

                selected_points = triggered["value"]["points"]
                selected_points = [point["pointIndex"] for point in selected_points]


                for target in targets:
                    #get index
                    target_idx = ...

                    # construct target callable args
                    args = ()
                    for arg in output["callable_args"]:

                        if arg == "selected_points":
                            args += selected_points

                        else:
                            # try to get variable from name?
                            args += ...

                    
                    # assign
                    output_repr[target_idx] = outputs["callable"](*args)

                return output_repr

            # double click and clear
            else:
f               or target in targets:
                    #get index
                    target_idx = ...

                    # construct target callable args
                    args = ()
                    for arg in output["callable_args"]:

                        if arg == "selected_points":
                            args += None

                        else:
                            # try to get variable from name?
                            args += ...

                    
                    # assign
                    output_repr[target_idx] = outputs["callable"](*args)

                return output_repr



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

