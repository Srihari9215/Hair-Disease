# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Hugging Face Spaces requires the application to listen on port 7860
ENV PORT 7860

# Use the Gunicorn command to start the app
# 'app:app' means the Flask app named 'app' inside the file 'app.py'
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 app:app