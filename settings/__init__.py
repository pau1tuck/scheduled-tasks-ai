# settings/__init__.py
import logging
import os

# from .celery import app as celery_app

# __all__ = ("celery_app",)

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

INTERNAL_IPS = [
    "127.0.0.1",
]

LOGGING_CONFIG = None
LOGGING = {
    "django.utils.autoreload": {
        "level": "INFO",
    }
}

# DEFINE THE ENVIRONMENT TYPE
PRODUCTION = STAGE = DEMO = LOCAL = False
dt_key = os.environ.get("DEPLOYMENT_TYPE", "LOCAL")
if dt_key == "PRODUCTION":
    PRODUCTION = True
    log_level = logging.INFO
elif dt_key == "DEMO":
    DEMO = True
    log_level = logging.INFO
elif dt_key == "STAGE":
    STAGE = True
    log_level = logging.DEBUG
else:
    LOCAL = True
    log_level = logging.INFO

logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

from settings.base import *
from settings.vendor import *

if LOCAL:
    from settings.local import *
