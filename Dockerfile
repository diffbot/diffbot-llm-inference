FROM vllm/vllm-openai:latest

# Install required packages
RUN apt-get update && apt-get install -y \
   supervisor \
   && apt-get clean

# Copy Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy diffbot-llm code
COPY . /code

WORKDIR /code

# Install requirements
RUN pip install poetry
RUN pip install pyasynchat # required by supervisord
RUN poetry env use python3.10
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN poetry run pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Expose ports
EXPOSE 3333 8000

# Start Supervisor
ENTRYPOINT ["/usr/bin/supervisord"]
