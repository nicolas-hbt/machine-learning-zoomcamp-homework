# Use the same slim Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the production requirements file
COPY requirements.txt .

# Install dependencies using standard pip
# This is the most reliable method in Docker.
# --no-cache-dir keeps the image layer small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model
# We need train.py because predict.py imports its 'preprocess' function
COPY ["predict.py", "train.py", "gb_model.pkl", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Define the command to run the application using gunicorn
# This will now work, as pip places 'gunicorn' on the default $PATH
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]