import gc
import os
import sys
import glfw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.viewer import Assignment1Viewer


if __name__ == "__main__":
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    try:
        Assignment1Viewer(
            width=1920,
            height=1080,
            title="HCMUT CO3059 - Assignment 1",
            distance=7.2,
        ).run()
    finally:
        gc.collect()
        glfw.terminate()