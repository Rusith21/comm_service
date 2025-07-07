FROM python:3.10-slim         # Base image
WORKDIR /app                  # Container working directory

COPY requirements.txt ./      # Copy dependencies list
RUN pip install --no-cache-dir -r requirements.txt

COPY . .                      # Copy app code and model
EXPOSE 8080                   # Port Cloud Run expects

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2"]
