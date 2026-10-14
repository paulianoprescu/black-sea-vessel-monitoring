import plotly.graph_objects as go
import requests
import pandas as pd
from plotly.subplots import make_subplots
import os


def load_portwatch(port_id, csv_path):
    """Fetches IMF Portwatch Port Calls.

    To be used to validate trends. 
    
    Args: 
        port_id: ID of the port we want to fetch information from
        csv_path: path where the output will be stored as csv
    
    Returns:
        A DataFrame with a date and a portcalls column, one row per day.

    Sources:
        https://www.w3schools.com/PYTHON/ref_requests_get.asp
        https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer/
        https://docs.python-requests.org/en/latest/user/quickstart/#json-response-content

    """
    # Fetch IMF Portwatch Constanta Port Calls 
    # To be used to validate trend 
    # Since there is a lot of systematic positive bias from the port area ship detection 
    # we are looking to validate the trend rather than absolute values
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["date"])
    url = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0/query"

    # ArcGIS API has a limit of 1000 lines per get, so we reset the request after 1000 gets.
    offset = 0
    pagesize = 1000

    # First count the amount of times port260 is in the database
    count = requests.get(url, params={"where": f"portid='{port_id}'", "returnCountOnly": "true", "f": "json"})
    total = count.json()["count"]

    rows = []
    while offset < total:
        params = {
            "where": f"portid='{port_id}'",
            "outFields": "year,month,day,portcalls",
            "orderByFields": "date",
            "resultOffset": offset,
            "resultRecordCount": pagesize,
            "f": "json",
        }

        data = requests.get(url, params=params).json()

        for f in data["features"]:
            rows.append(f["attributes"])
        offset += pagesize

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df.drop(columns=["year", "month", "day"], inplace=True)
    
    # Save port calls to local directory 
    df.to_csv(csv_path, index=False)
    return df


def monthly_means(ships, portwatch, start_date, end_date):
    """Calculates the monthly means of detected ships and PortWatch port calls.
    
    Args:
        ships: DataFrame with 'date', 'port', 'anchorage' and 'total' columns
        portwatch: DataFrame with 'date' and 'portcalls' columns
        start_date: first date included, e.g. '2019-01-01'
        end_date: first datea not included, e.g. '2026-01-01

    Returns: 
        monthly: DataFrame with one row per month and a column per series
        corr: Spearman correlation of each series with the port calls

    Sources:
        https://www.geeksforgeeks.org/python/how-to-filter-dataframe-rows-based-on-the-date-in-pandas/
        https://stackoverflow.com/questions/65471540/get-monthly-average-in-pandas
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
    """

    # Convert dates and use the same end date for both datasets
    portwatch['date'] = pd.to_datetime(portwatch['date'])
    ships['date'] = pd.to_datetime(ships['date'])
    portwatch = portwatch.loc[(portwatch['date'] >= start_date) & (portwatch['date'] < end_date)]

    # Resample and split different detections onto port, anchorage and total area
    # Take monthly means, so both datasets have the same time steps
    pw_months_mean = portwatch.resample('MS', on='date')['portcalls'].mean()
    total_predict_mean = ships.resample('MS', on='date')['total'].mean()
    port_predict_mean = ships.resample('MS', on='date')['port'].mean()
    anchorage_predict_mean = ships.resample('MS', on='date')['anchorage'].mean()

    # One table with a column per series
    monthly = pd.concat([pw_months_mean, total_predict_mean, port_predict_mean, anchorage_predict_mean], axis=1)
    
    # Spearman correlation of each detection series with the port calls
    corr = monthly.corr(method='spearman')['portcalls']
    
    return (monthly, corr)

def plot_comparisons(monthly, corr, events):
    """Plot detected ships and PortWatch port calls per month.

    Args:
        monthly: DataFrame from monthly_means
        corr: Spearman correlations from monthly_means
        events: dict of {label: date} drawn as dashed lines

    Returns: 
        A plotly figure with a dropdown for total, port and anchorage
    
    Sources:
        https://plotly.com/python/graph-objects/
        https://plotly.com/python/horizontal-vertical-shapes/

    """
    # Figure with a second y-axis on the right for the port calls
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_yaxes(title_text="Detected ships (monthly mean per scene)", secondary_y=False)
    fig.update_yaxes(title_text="PortWatch port calls (monthly mean per day)", secondary_y=True)

    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["total"], name=f"Total (ρ = {corr['total']:.2f})"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["port"], name=f"Port (ρ = {corr['port']:.2f})", visible=False), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["anchorage"], name=f"Anchorage (ρ = {corr['anchorage']:.2f})", visible=False), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["portcalls"], name="IMF Portwatch Port Calls"), secondary_y=True)

    # Toggle buttons to switch between total, port and anchorage
    buttons = [
        dict(label="Total", method="update",
            args=[{"visible": [True, False, False, True]}]),
        dict(label="Port", method="update",
            args=[{"visible": [False, True, False, True]}]),
        dict(label="Anchorage", method="update",
            args=[{"visible": [False, False, True, True]}]),
    ]
    fig.update_layout(updatemenus=[dict(buttons=buttons)])

    # Vertical lines for the important events
    # Dates in milliseconds, because add_vline gives an error with a date and a label    
    for label, date in events.items():
        fig.add_vline(x=pd.Timestamp(date).timestamp() * 1000, line_dash='dash',
                annotation_text=label)

    return fig