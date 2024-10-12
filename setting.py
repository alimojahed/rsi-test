import logging
import argparse

class CustomFormatter(logging.Formatter):

    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[92m',    # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m' # Magenta
    }
    RESET = '\033[0m'  # Reset color

    def format(self, record):
        log_msg = super().format(record)

        color = self.COLORS.get(record.levelname, self.RESET)
        log_msg = f"{color}{log_msg}{self.RESET}"

        return log_msg

logger = logging.getLogger('logger')

logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()

formatter = CustomFormatter('[%(levelname)s] - %(asctime)s - %(message)s', datefmt='%Y-%m-%d-%H-%M-%S')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


parser = argparse.ArgumentParser(description="This is the test program for RSI")

parser.add_argument("--qrcode", type=str, default="images/photo_2024-10-10_00-24-24.jpg", help="path to qrcode image")
parser.add_argument("--debug", action="store_true", help="enable debug mode to see step by step results")
parser.add_argument("--model", type=str, default="models/best-yolov11n-cls.pt", help="Yolo classifier model")