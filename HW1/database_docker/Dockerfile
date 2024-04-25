# Use a Python base image
FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Copy the script and CSV file into the container
COPY database_script.ipynb .
COPY heart_disease.csv .

# Install MySQL client library
# RUN apt-get update && apt-get install -y mysql-client

RUN pip install mysql pandas mysql-connector-python jupyter sqlalchemy matplotlib scipy numpy

EXPOSE 8888

# Start Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
