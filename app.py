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

# Cargar variables de entorno
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Cargar el modelo preentrenado
model = joblib.load("xgboost_model.pkl")

# Cargar la imagen de la curva ROC
with open("roc_curve.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

def get_connection():
    """Establece una conexión con la base de datos."""
    return psycopg2.connect(DATABASE_URL)

def fetch_data(query):
    """Ejecuta una consulta SQL y devuelve los resultados como DataFrame."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=columns)

# Inicializar la app Dash
app = dash.Dash(__name__)
server = app.server  # Exponer el objeto Flask

# Variables globales para navegación
current_index = 0

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Polimeromics Dashboard Prediction Algorithm", style={'textAlign': 'center'}),

    # Sección de predicción del modelo
    html.Div([
        html.H3("Test XGBoost Model"),
        html.Button("Show Metrics", id="show-metrics", n_clicks=0),
        html.Div(id="xgboost-metrics", style={'marginTop': 20}),
        html.Img(src=f"data:image/png;base64,{encoded_image}", style={'width': '100%', 'height': 'auto', 'marginTop': 20}),
    ], style={'width': '50%', 'margin': '0 auto', 'textAlign': 'center'}),

    # Sección de consulta a la base de datos
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

# Callback para mostrar las métricas preentrenadas
@app.callback(
    Output("xgboost-metrics", "children"),
    [Input("show-metrics", "n_clicks")]
)
def show_pretrained_metrics(n_clicks):
    if n_clicks == 0:
        return ""

    metrics = [
        html.P("Accuracy: 0.93"),
        html.P("F1-Score: 0.91"),
        html.P("AUC: 0.95")
    ]

    return metrics

# Callback para mostrar el número total de registros en la base de datos seleccionada
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

# Callback para navegar por los registros de la base de datos seleccionada
@app.callback(
    Output('record-output', 'children'),
    [Input('prev-record', 'n_clicks'), Input('next-record', 'n_clicks'), Input('db-selector', 'value')]
)
def update_record(prev_clicks, next_clicks, db_value):
    global current_index

    if db_value is None:
        return "Please select a database."

    try:
        # Consulta los registros de la tabla seleccionada
        query = f"SELECT * FROM {db_value} LIMIT 100;"
        data = fetch_data(query)

        # Manejo de navegación
        triggered = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if triggered == "next-record" and next_clicks > 0:
            current_index = min(current_index + 1, len(data) - 1)
        elif triggered == "prev-record" and prev_clicks > 0:
            current_index = max(current_index - 1, 0)

        # Mostrar el registro actual
        record = data.iloc[current_index]
        return html.Pre("\n".join([f"{col}: {val}" for col, val in record.items()]))

    except Exception as e:
        return f"Error retrieving records: {e}"

# Ejecutar el servidor
if __name__ == "__main__":
    app.run_server(debug=True)


