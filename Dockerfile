# Install python3 and pip3 (from preexisting image)
FROM python:3.7

# Create virtual environment:
RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy source cody
WORKDIR /app
ADD . /app

# Install dependencies:
RUN pip3 install -r requirements.txt

# Expose application port:
EXPOSE 5002

# Run the application:
CMD ["python", "app.py"]
