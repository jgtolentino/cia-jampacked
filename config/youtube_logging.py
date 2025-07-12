import logging
import os
from datetime import datetime

def setup_logging():
    """Setup logging for YouTube integration"""
    log_dir = "/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"youtube_integration_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('youtube_integration')
