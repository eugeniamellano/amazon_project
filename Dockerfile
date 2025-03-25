# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install pandas

# Make a volume mount point for the input/output CSV files
VOLUME ["/app/input_data.csv", "/app/output_data.csv"]

# Run the application (by default, run the main ETL process)
CMD ["python", "etl.py"]
