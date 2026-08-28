FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# unprivileged
RUN useradd -m -u 10001 proxy
USER proxy

COPY config/example.yaml /etc/llm-proxy/example.yaml

ENV CONFIG=/etc/llm-proxy/prod.yaml
EXPOSE 4000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:4000/health', timeout=2).status==200 else 1)"

ENTRYPOINT ["llm-proxy"]
CMD ["--config", "/etc/llm-proxy/prod.yaml"]
