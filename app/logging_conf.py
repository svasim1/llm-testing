import os
import logging
from datetime import datetime

# Setup logging
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_directory, log_filename), encoding="utf-8"),
        #logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)