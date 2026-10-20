FROM python:3.12-slim

WORKDIR /app
COPY client/requirements.txt /app/client/requirements.txt
RUN pip install --no-cache-dir -r /app/client/requirements.txt

COPY proto/ /app/proto/
COPY client/inference_client.py /app/client/
# Generate the gRPC stubs at build time.
RUN python -c "import sys; sys.path.insert(0, '/app/client'); import inference_client"

ENV MIE_SERVER=inference-server:50051
ENTRYPOINT ["python", "/app/client/inference_client.py"]
CMD ["demo"]
