FROM python:3.12-slim
# Set the working directory
WORKDIR /app

#Create model directory
RUN mkdir -p /app/model
#create Schema directory
RUN mkdir -p /app/dataset

# Copy the dataset into the container at /app/dataset
COPY dataset/*.csv /app/dataset/
# Copy the requirements file into the container at /app
COPY app/requirements.txt .
# Copy trained model into the container at /app/model
COPY trained_model/xgboost_model.pkl /app/model/
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code into the container at /app
COPY app/main.py .
# Expose the port the app runs on
EXPOSE 8001
# Run the application
CMD ["python", "main.py"]