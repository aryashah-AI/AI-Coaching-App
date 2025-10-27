import logging
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), '../../logs/base.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("TripleJumpApp")
