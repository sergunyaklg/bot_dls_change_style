# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Add current directory files to the docker container
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install aiogram && \
    pip install torch torchvision && \
    pip install pillow && \
    pip install matplotlib

# Define environment variable
ENV NAME World

CMD ["python", "app.py"]

# docker build -t telegram-bot .

# docker run --env-file env.list -p 4000:80 your-image-name
