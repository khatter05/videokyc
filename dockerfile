# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Start both FastAPI and Streamlit using supervisord
CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 & streamlit run stn.py --server.port 8501 --server.address 0.0.0.0"]
