FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Healthcheck wymaga wget + dodaj sqlite3
RUN apt-get update && apt-get install -y --no-install-recommends wget sqlite3 && rm -rf /var/lib/apt/lists/*

COPY ./templates /app/templates/
COPY . .
RUN python - << 'PY'
import compileall, sys
ok = compileall.compile_dir('/app', maxlevels=10, quiet=1)
sys.exit(0 if ok else 1)
PY

EXPOSE 5000

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=UTF-8
CMD ["python", "main.py"]
