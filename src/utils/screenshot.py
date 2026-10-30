import datetime
import os
import glfw
import numpy as np
import OpenGL.GL as GL
from PIL import Image


def save_framebuffer_screenshot(win, output_dir="screenshots", prefix="capture"):
    """Reads the current OpenGL framebuffer and saves it as a PNG file."""
    os.makedirs(output_dir, exist_ok=True)
    width, height = glfw.get_framebuffer_size(win)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.png")

    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    img_flipped = np.flipud(img)

    Image.fromarray(img_flipped, mode="RGBA").save(filepath)
    return filepath