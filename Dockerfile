FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN python -m pip install --upgrade pip

# Copiar los archivos del proyecto
COPY requirements.txt /app/requirements.txt

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar TensorFlow y dependencias
RUN pip install tensorflow==2.10.0 tensorboard==2.10.0 tensorflow-io-gcs-filesystem==0.27.0

# Instalar dependencias del proyecto
RUN pip install -r requirements.txt

# Copiar el resto de los archivos
COPY . /app/

# Ejecutar la aplicación
CMD ["python", "app.py"]

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
