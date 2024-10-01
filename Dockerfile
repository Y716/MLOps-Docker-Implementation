# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Verify installations
RUN python -c "import numpy; print('numpy version:', numpy.__version__)"
RUN python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the default command to run the app
CMD ["python", "app.py"]
