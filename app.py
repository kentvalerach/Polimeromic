import dash
import base64
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import joblib
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Load the pretrained model
model = joblib.load("xgboost_model.pkl")

# Load the ROC curve image
with open("roc_curve.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

def get_connection():
    """Establish a connection to the database."""
    return psycopg2.connect(DATABASE_URL)

def fetch_data(query):
    """Execute an SQL query and return the results as a DataFrame."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=columns)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the Flask server

# Global variables for navigation
current_index = 0

# Layout of the application
app.layout = html.Div([
    html.H1("Polimeromics Data Explorer", style={'textAlign': 'center'}),

    # Left section: ROC curve and metrics
    html.Div([
        html.H3("Model XGboost Metrics"),
        html.Img(src=f"data:image/png;base64,{encoded_image}", style={'width': '100%', 'height': 'auto', 'marginTop': 20}),
        html.Div([
            html.Pre("Classification Report:\n\n"
                     "Precision: 0.99 (Class 0), 0.98 (Class 1)\n"
                     "Recall: 0.98 (Class 0), 0.99 (Class 1)\n"
                     "F1-Score: 0.99 (Class 0), 0.98 (Class 1)\n\n"
                     "Accuracy: 0.99\n"
                     "Macro Avg: Precision 0.99, Recall 0.99, F1-Score 0.99\n"
                     "Weighted Avg: Precision 0.99, Recall 0.99, F1-Score 0.99"\n\n"
                     "The metrics shown are the result of a pre-trained XGboost Machine Learning XGboost model that has been uploaded to\n" 
                     "this dashboard.You can access the Python script in the repository https://github.com/kentvalerach/Polimeromic  the\n" 
                     "results shown is a Big data transformation and cleaning process applied to biochemical downloaded from:\n"
                     "https://www.rcsb.org/  study data: RCSB_PDB_Macromolecular_Structure_Datasetand from  https://thebiogrid.org\n"
                     "study data BIOGRID-ORCS-ALL1-homo_sapiens-1.1.16.screens.\n"
                     "This is an example of bioinformatics to be applied in scientific studies and laboratory tests.")
        ], style={'marginTop': 20, 'textAlign': 'left'}),
    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

    # Right section: Database query
    html.Div([
        html.H3("Database Query"),
        dcc.Dropdown(
            id='db-selector',
            options=[
                {'label': 'Biogrid Homo Sapiens', 'value': 'biogrid_homosapiens'},
                {'label': 'RCSB PDB', 'value': 'rcsb_pdb'}
            ],
            placeholder="Select a database"
        ),
        html.Div(id="db-records-count", style={'marginTop': 10, 'fontWeight': 'bold'}),
        html.Button("Previous", id="prev-record", n_clicks=0),
        html.Button("Next", id="next-record", n_clicks=0),
        html.Div(id="record-output", style={'marginTop': 20}),
    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'}),
])

# Callback to show total records in the selected database
@app.callback(
    Output('db-records-count', 'children'),
    [Input('db-selector', 'value')]
)
def update_db_record_count(db_value):
    if db_value is None:
        return "Select a database to see the total number of records."
    try:
        query = f"SELECT COUNT(*) FROM {db_value};"
        count = fetch_data(query).iloc[0, 0]
        return f"Total number of records: {count}"
    except Exception as e:
        return f"Error accessing the database: {e}"

# Callback to navigate through records in the selected database
@app.callback(
    Output('record-output', 'children'),
    [Input('prev-record', 'n_clicks'), Input('next-record', 'n_clicks'), Input('db-selector', 'value')]
)
def update_record(prev_clicks, next_clicks, db_value):
    global current_index

    if db_value is None:
        return "Please select a database."

    try:
        # Query records from the selected table
        query = f"SELECT * FROM {db_value} LIMIT 100;"
        data = fetch_data(query)

        # Handle navigation
        triggered = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if triggered == "next-record" and next_clicks > 0:
            current_index = min(current_index + 1, len(data) - 1)
        elif triggered == "prev-record" and prev_clicks > 0:
            current_index = max(current_index - 1, 0)

        # Display the current record
        record = data.iloc[current_index]
        return html.Pre("\n".join([f"{col}: {val}" for col, val in record.items()]))

    except Exception as e:
        return f"Error retrieving records: {e}"

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)

