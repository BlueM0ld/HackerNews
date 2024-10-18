# Base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model script to the container
COPY model_hk_titles.py .

# Copy the JSON files and the model file
COPY words_to_ids.json .
COPY ids_to_words.json .
COPY skipgram_model.pth .

EXPOSE 8000

# Define the command to run your script
# CMD ["python", "model_hk_titles.py"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
