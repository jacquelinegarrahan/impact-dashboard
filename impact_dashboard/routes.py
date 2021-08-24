
"""Routes for parent Flask app."""
from flask import current_app as app
from flask import Flask, render_template, request
import pandas as pd
import json
from pymongo import MongoClient
from dateutil import parser
import plotly
import plotly.express as px
import os


MONGO_HOST = os.environ["MONGO_HOST"]
MONGO_PORT = int(os.environ["MONGO_PORT"])

CLIENT = MongoClient(MONGO_HOST, MONGO_PORT)
DB = CLIENT.impact
RESULTS = DB.results

ALL_PVS = []
for res in RESULTS.find():
    ALL_PVS += res["inputs"].keys()
ALL_PVS=set(ALL_PVS)

@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.jinja2",
        title="Plotly Dash Flask Tutorial",
        description="Embed Plotly Dash into your Flask applications.",
        template="home-template",
        body="This is a homepage served with Flask.",
    )

@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return get_pv(request.args.get('data'))
   


def get_pv(pv="distgen:n_particle"):

    if pv:
        results = list(RESULTS.find())
        data = [res["inputs"][pv] for res in results]
        isotime = [res["isotime"] for res in results]

    df = pd.DataFrame({pv: data, "date": isotime})
    df["date"] = pd.to_datetime(df["date"])

    fig = px.line(df, x="date", y=pv)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON