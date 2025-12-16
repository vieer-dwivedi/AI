import logging, os
from app.config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "executor.log")),
        logging.StreamHandler()
    ],
)
