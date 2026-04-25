FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY controller.py /app/controller.py
COPY failure_modes.py /app/failure_modes.py

EXPOSE 6660

CMD ["uvicorn", "controller:app", "--host", "0.0.0.0", "--port", "6660"]

