import logging
from app.core.config import settings

def setup_logging():
    log_level = settings.LOG_LEVEL.upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Uvicorn's access logs can be quite verbose on their own.
    # If you want to reduce FastAPI/Uvicorn default logging noise:
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING) # or ERROR
    logging.getLogger("fastapi").setLevel(logging.INFO) # Or your app's log level