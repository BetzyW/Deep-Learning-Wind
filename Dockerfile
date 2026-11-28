# Entorno reproducible del proyecto (CPU). Construir y correr:
#   docker build -t viento-dl .
#   docker run --rm viento-dl                       # pruebas unitarias y construcción del libro
#   docker run --rm -v "$PWD/datos:/proyecto/datos" -v "$PWD/resultados:/proyecto/resultados" viento-dl #       python src/estadistica_e3.py                 # análisis completo, con los datos y las predicciones montados
# Los entrenamientos se hicieron con GPU (CUDA 12.6). Se versionan las métricas por corrida (CSV) y las tablas; los datos
# y las predicciones por ventana (.npz) no se versionan por tamaño y licencia: ver README.md y
# experiments/conjuntos_adicionales.md.
FROM python:3.11-slim
WORKDIR /proyecto
ENV PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt req/proyecto.txt
COPY book/requirements.txt req/libro.txt
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && grep -v -E "^(torch|timeseries-cv|jupyter)" req/proyecto.txt > req/sin_torch.txt \
 && pip install --no-cache-dir -r req/sin_torch.txt onnx onnxruntime pytest arch==8.0.0 \
 && pip install --no-cache-dir --no-deps timeseries-cv==0.1.5 \
 && pip install --no-cache-dir -r req/libro.txt
COPY . .
CMD ["sh", "-c", "pytest -q tests && jupyter-book build book/"]
