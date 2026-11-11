FROM mcr.microsoft.com/devcontainers/python:3.12

RUN mkdir -p /home/vscode/.cache && chown -R vscode:vscode /home/vscode/.cache

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt
