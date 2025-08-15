# Simple, fast image for Python apps
FROM python:3.11-slim

# HF Spaces recommend running as uid 1000 and setting a working dir
# (avoids permissions issues and enables Dev Mode cleanly)
# Docs: Docker Spaces "Permissions" section
# https://huggingface.co/docs/hub/en/spaces-sdks-docker
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install Python deps first (better layer caching)
COPY --chown=user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user . $HOME/app

# Streamlit will serve the app; Spaces default external port is 7860
# You can also set --server.port=$PORT (with shell form) but we fix it to 7860
# to match README's app_port.
EXPOSE 7860

# Disable telemetry and tighten server settings for Spaces
# (headless, no CORS/XSRF issues behind HF reverse proxy)
CMD ["bash","-lc","streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false"]
