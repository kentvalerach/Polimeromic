FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && apt-get clean

# Instalar Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Actualizar pip e instalar dependencias
RUN python -m pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Copiar el resto de los archivos
COPY . /app

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
