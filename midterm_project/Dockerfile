# Use the same slim Python 3.12 base image
FROM python:3.12-slim

# Install uv, the tool we use for package management
RUN pip install uv

# Set the working directory inside the container
WORKDIR /app

# Copy the production requirements file
COPY requirements.txt .

# Install the dependencies using uv
# mind the --system as we want to install them globally in the container, otherwise it won't work
# uv is designed to protect your system by refusing to install packages globally (outside a virtual environment) by default
RUN uv pip install --system -r requirements.txt

# Copy the application code and the trained model
# We need train.py because predict.py imports its 'preprocess' function
COPY ["predict.py", "train.py", "gb_model.pkl", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Define the command to run the application using gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]