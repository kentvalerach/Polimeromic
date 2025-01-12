import dash
import base64
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import psycopg2
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import os  

# Carga las variables desde el archivo .env
load_dotenv()

# Accede a las variables de entorno
DATABASE_URL = os.getenv("DATABASE_URL")

DEBUG = os.getenv("DEBUG", "False")  # Por defecto, "False" si no está definido


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

# Variables globales para navegación
current_index = 0
selected_db = None

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Polimeromics Dashboard Prediction Algorithm", style={'textAlign': 'center'}),

    # Sección de predicción del modelo
    html.Div([
        html.H3("Test XGBoost Model"),
        html.Button("Run XGBoost Model", id="test-xgboost", n_clicks=0),
        html.Div(id="xgboost-metrics", style={'marginTop': 20}),
        dcc.Graph(id="auc-plot"),
    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),

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

    # Sección para subir archivos CSV
    html.Div([
    html.H3("Upload CSV for Prediction"),
    html.P("Columns required:"),
    html.Ul([
        html.Li("score_1 (float)"),
        html.Li("score_2 (float)"),
        html.Li("molecular_weight (float)"),
        html.Li("aliases (text)"),
        html.Li("ligand_mw (float)")
    ]),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag or Select a CSV File (max 50MB)']),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='upload-output', style={'marginTop': 20}),
], style={'width': '50%', 'margin': '0 auto'}),

])

    

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

# Callback para probar el modelo XGBoost
@app.callback(
    [Output("xgboost-metrics", "children"),
     Output("auc-plot", "figure")],
    [Input("test-xgboost", "n_clicks")]
)
def test_xgboost_model(n_clicks):
    if n_clicks == 0:
        return "", go.Figure()

    try:
        # Cargar y combinar datos desde PostgreSQL
        biogrid_query = "SELECT * FROM biogrid_homosapiens;"
        rcsb_query = "SELECT * FROM rcsb_pdb;"
        biogrid_df = fetch_data(biogrid_query)
        rcsb_df = fetch_data(rcsb_query)

        combined_data = pd.merge(
            biogrid_df, rcsb_df,
            left_on='official_symbol',
            right_on='macromolecule_name',
            how='inner'
        )

        # Crear nuevas características
        combined_data["score_product"] = combined_data["score_1"] * combined_data["score_2"]
        combined_data["aliases_length"] = combined_data["aliases"].str.len()
        combined_data["ligand_mw_normalized"] = combined_data["ligand_mw"] / combined_data["molecular_weight"]

        # Seleccionar características y variable objetivo
        features = ["score_1", "score_2", "score_product", "aliases_length", "ligand_mw_normalized"]
        X = combined_data[features].fillna(0)
        y = combined_data["hit"].apply(lambda x: 1 if x == "YES" else 0)

        # Balancear clases usando SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

        # Inicializar el modelo
        model = XGBClassifier(
            max_depth=3,
            min_child_weight=5,
            reg_lambda=1,
            reg_alpha=1,
            random_state=42,
            eval_metric="logloss"
        )

        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        metrics = [
            html.P(f"Accuracy: {accuracy:.2f}"),
            html.P(f"F1-Score: {f1:.2f}"),
            html.P(f"Area Under Curve (AUC): {auc:.2f}"),
        ]

        # Visualizar AUC
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc_fig = go.Figure()
        auc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC'))
        auc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        auc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

        # Guardar el modelo
        joblib.dump(model, "models/xgboost_model.pkl")

        return metrics, auc_fig

    except Exception as e:
        return [html.P(f"Error testing the model: {e}")], go.Figure()

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

@app.callback(
    Output('upload-output', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def process_uploaded_file(contents, filename):
    if contents is None:
        return "Please upload a CSV file."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Leer el archivo CSV
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Verificar columnas requeridas
        required_columns = ["score_1", "score_2", "molecular_weight", "aliases", "ligand_mw"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return html.Div([
                html.P(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}"),
            ])

        # Mostrar una vista previa de las primeras filas
        return html.Div([
            html.P(f"File {filename} uploaded successfully!"),
            html.Pre(df.head().to_string(index=False))
        ])

    except Exception as e:
        return html.P(f"Error processing the file: {e}")


# Ejecutar el servidor



server = app.server  # Esto expone el servidor de Flask que Gunicorn busca

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)



