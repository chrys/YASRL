import multiprocessing
import os

# Server socket
bind = "unix:/run/yasrl-api/yasrl-api.sock"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Cap at 8 workers
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300
graceful_timeout = 30
keepalive = 5

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 100

# Process naming
proc_name = "yasrl-api"

# Daemon mode - let systemd handle this
daemon = False

# User/group to run as
user = "deploy"
group = "www-data"

# PID file - only used for FastAPI/uvicorn worker
pidfile = "/run/yasrl-api/yasrl-api.pid"

# Note: This config is for API only. Flask UI uses its own systemd service without pidfile.

# Preload application
preload_app = True

# Working directory
chdir = "/srv/library2"

# Python path
pythonpath = "/srv/library2/src"

# Environment variables
raw_env = [
    "PYTHONPATH=/srv/library2/src",
]

# Logging - send to stdout/stderr for journald
errorlog = "-"
accesslog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Worker process lifecycle hooks
def on_starting(server):
    server.log.info("YASRL API server starting up")

def on_reload(server):
    server.log.info("YASRL API server reloading")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def post_fork(server, worker):
    server.log.info(f"Worker initialized (pid: {worker.pid})")

def worker_abort(worker):
    worker.log.error(f"Worker aborted (pid: {worker.pid})")