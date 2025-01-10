import os
import subprocess
import sys

def run_command(command):
    """Ejecuta un comando en la terminal."""
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando el comando: {command}")
        sys.exit(1)

def main():
    print("=== Configuración del Proyecto ===")

    # Paso 1: Crear entorno virtual
    print("\n1. Creando el entorno virtual...")
    if not os.path.exists("venv"):
        run_command(f"{sys.executable} -m venv venv")
    else:
        print("El entorno virtual ya existe. Saltando este paso.")

    # Paso 2: Activar entorno virtual
    print("\n2. Activando el entorno virtual...")
    if os.name == "nt":  # Windows
        activate_command = ".\\venv\\Scripts\\activate"
    else:  # macOS/Linux
        activate_command = "source venv/bin/activate"
    print(f"Ejecuta este comando manualmente para activar el entorno: {activate_command}")

    # Paso 3: Instalar dependencias
    print("\n3. Instalando dependencias...")
    requirements = [
        "dash",
        "pandas",
        "xgboost",
        "scikit-learn",
        "psycopg2-binary",
        "jupyter",  # Opcional: Para usar Jupyter Notebook
        "ipykernel",  # Opcional: Para registrar el entorno en Jupyter
    ]
    for package in requirements:
        print(f"Instalando {package}...")
        run_command(f"venv\\Scripts\\pip install {package}")

    # Paso 4: Registrar el entorno en Jupyter (opcional)
    print("\n4. Registrando el entorno en Jupyter Notebook...")
    run_command("venv\\Scripts\\python -m ipykernel install --user --name=venv --display-name 'Python (venv)'")

    print("\n=== Configuración Completada ===")
    print("Para usar este entorno, actívalo con el siguiente comando:")
    print(f"    {activate_command}")
    print("Luego, ejecuta tu script con:")
    print("    python app.py")

if __name__ == "__main__":
    main()
