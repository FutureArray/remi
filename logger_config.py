import os
import logging
from datetime import datetime

def setup_logging(output_folder):
    # 创建一个包含时间信息的日志文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_folder, f'finetune_{current_time}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return logging.getLogger(__name__)

def get_logger():
    return logging.getLogger(__name__)