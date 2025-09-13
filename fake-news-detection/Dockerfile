# Dockerfile for Fake News Detection App
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app/requirements.txt .
COPY requirements.txt ./base_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r base_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "streamlit" ]; then\n\
    streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0\n\
elif [ "$1" = "api" ]; then\n\
    uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000\n\
else\n\
    echo "Usage: docker run <image> [streamlit|api]"\n\
    exit 1\n\
fi' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]