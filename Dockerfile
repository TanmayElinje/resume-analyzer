# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Create the cache directory and set its permissions
RUN mkdir -p /app/cache && chown -R 1000:1000 /app/cache

# Set the cache directory for Hugging Face models to a writable location
ENV HF_HOME=/app/cache

# Copy your requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that Hugging Face Spaces uses
EXPOSE 7860

# The command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--preload", "app:app"]