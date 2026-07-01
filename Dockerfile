FROM ilkaybrahim/deepstream_app:8.0
COPY ./server/backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "backend.app.app:app", "--host", "0.0.0.0", "--port", "8000"]