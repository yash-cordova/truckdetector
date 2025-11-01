"""Main entry point for Dock Monitoring Application"""
import sys
import warnings
import os

# Fix FFmpeg async lock errors by setting environment before importing OpenCV
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_INTEL_MFX"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_BACKEND"] = "CAP_ANY"
os.environ["OPENCV_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)

from PyQt5.QtWidgets import QApplication
from window import MainWindow


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

