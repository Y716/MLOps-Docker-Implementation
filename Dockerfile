# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install git and build tools
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Define the repository to clone
ENV REPO_URL=https://github.com/Y716/MLOps-Docker-Implementation.git
ENV APP_DIR=/app

# Clone the repository
RUN git clone $REPO_URL $APP_DIR

# Set the working directory
WORKDIR $APP_DIR

# Ensure the models directory exists
RUN mkdir -p models

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define the default command to run the app
CMD ["python", "app.py"]
