FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install Python deps, then Playwright system libs + Chromium
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install-deps chromium \
    && playwright install chromium

COPY . .

ENV PYTHONUNBUFFERED=1
ENV UV_THREADPOOL_SIZE=1

CMD ["python", "server.py"]
