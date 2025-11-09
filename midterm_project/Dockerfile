# Stage 1: Build the environment
# Use the Python version specified in pyproject.toml
FROM python:3.11-slim as builder

# Install uv (the new fast pip)
RUN pip install uv

WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies using the lock file for reproducible builds
RUN uv pip sync uv.lock

# ---
# Stage 2: Create the final, lightweight production image
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment (with installed packages) from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy the application source code and model artifacts
# IMPORTANT: You must run train.py to create these files BEFORE building!
COPY serve.py .
COPY utils.py .
COPY processor.pkl .
COPY model.pkl .

# Expose the port the app will run on
EXPOSE 9696

# Define the command to run the application using gunicorn
# This runs 4 worker processes, binding to all interfaces on port 9696
CMD ["gunicorn", "--bind=0.0.0.0:9696", "--workers=4", "serve:app"]