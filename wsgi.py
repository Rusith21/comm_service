# wsgi.py
from app import create_app

# WSGI entry point for Gunicorn
app = create_app()
