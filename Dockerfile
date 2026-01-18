FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# local use
CMD ["python", "app.py"]


#production
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]