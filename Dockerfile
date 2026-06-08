# Hugging Face Spaces (Docker SDK) — runs the Streamlit app.
# Full python image (not slim) so onnxruntime/chromadb find libgomp etc.
FROM python:3.12

# HF Spaces run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache

WORKDIR $HOME/app

# Install deps first for better layer caching.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY --chown=user . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
