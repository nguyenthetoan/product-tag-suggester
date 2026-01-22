"""Gunicorn configuration for production-like stability."""

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"

# Worker processes
# Use 1 worker for ML models to avoid loading model multiple times
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout - ML inference can take a while
timeout = 120
graceful_timeout = 30

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "product-tag-suggester"

# Restart workers after this many requests (prevents memory leaks)
max_requests = 100
max_requests_jitter = 10
