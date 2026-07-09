# syntax=docker/dockerfile:1

# Immagine base slim (Python 3.12): runtime leggero.
# 3.12 richiesta da numpy==2.5.1 nel requirements.lock (necessita Python >= 3.12).
FROM python:3.12-slim

# Niente .pyc su disco, output non bufferizzato (log immediati),
# Streamlit headless (nessun prompt interattivo all'avvio in container).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Utente non-root dedicato (il processo non gira da root).
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Dipendenze PRIMA del codice: sfrutta il layer cache di Docker
# (le dipendenze cambiano di rado, il codice spesso).
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Codice applicativo con ownership ad appuser.
COPY --chown=appuser:appuser . .

# Passa all'utente non privilegiato.
USER appuser

# Porta del server Streamlit.
EXPOSE 8501

# Healthcheck sull'endpoint interno di Streamlit (extra production-grade).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

# Avvio del frontend.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
