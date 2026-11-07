# GPU image for training and evaluation. Start from the verl image that matches verl 0.9.1 (see verl's
# docs for the tag), then add this repository. SWE-bench evaluation additionally needs access to a Docker
# daemon (mount /var/run/docker.sock).
ARG BASE=verlai/verl:latest
FROM ${BASE}
WORKDIR /workspace/CHOP
COPY . .
RUN pip install --no-cache-dir -r environment/requirements-gpu.txt && pip install --no-cache-dir -e .
