
from ipycanvas import Canvas
from PIL import Image
import numpy as np
import time
import os

import cv2

image_path = 'assets/scene_01.png'
# Use an example image path or replace with your own
save_path = 'current_canvas.png'


is_mouse_down = False
position_at_mouse_down = None
position_at_mouse_up = None
image = None
original_image = None
scale = 1
canvas = Canvas(width=256, height=256, sync_image_data=True)
processed_image = None

def draw_rectangle(x0, y0, x1, y1, color='red', fill=False):
    x1 = x0 + ((x1 - x0)//(16 * scale)) * 16 * scale
    y1 = y0 + ((y1 - y0)//(16 * scale)) * 16 * scale
    canvas.stroke_style = color
    canvas.line_width = 3
    canvas.stroke_rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
    if fill:
        canvas.fill_style = color
        canvas.fill_rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        
# Draw with red brush when dragging
def handle_mouse_down(x, y):
    global is_mouse_down, position_at_mouse_down
    is_mouse_down = True
    position_at_mouse_down = (x, y)
    
def handle_mouse_up(x, y):
    global is_mouse_down, position_at_mouse_up, image, processed_image
    is_mouse_down = False
    position_at_mouse_up = (x, y)
    if position_at_mouse_down is not None:
        # Draw the rectangle directly on the image array and update the canvas
        x0, y0 = position_at_mouse_down
        x1, y1 = x, y

        X0 = int(x0 / scale)
        Y0 = int(y0 / scale)
        X1 = X0 + ((x1 - x0) / scale //16) * 16
        Y1 = Y0 + ((y1 - y0) / scale //16) * 16
        
        # Crop the image
        processed_image = original_image.crop((X0, Y0, X1, Y1))

        processed_image.save(save_path)




def handle_mouse_move(x, y):
    global update_frequency, last_update
    if not is_mouse_down:
        return

    # Draw rectangle from position_at_mouse_down to current (x, y)
    if position_at_mouse_down is not None:
        # Clear and redraw the image
        canvas.put_image_data(np.array(image))
        draw_rectangle(position_at_mouse_down[0], position_at_mouse_down[1], x, y)
        pass

# canvas.observe(save_to_file, "image_data")
canvas.on_mouse_down(handle_mouse_down)
canvas.on_mouse_move(handle_mouse_move)
canvas.on_mouse_up(handle_mouse_up)


def crop_image(image_path, _save_path = 'current_canvas.png', force=False):
    global image, save_path, original_image, scale, processed_image
    save_path = _save_path
    original_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_image.size
    max_side = 1024
    if orig_w >= orig_h:
        new_w = max_side
        new_h = int(orig_h * max_side / orig_w)
    else:
        new_h = max_side
        new_w = int(orig_w * max_side / orig_h)
    
    original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
    orig_w, orig_h = original_image.size

    max_side = 512
    if orig_w >= orig_h:
        new_w = max_side
        new_h = int(orig_h * max_side / orig_w)
    else:
        new_h = max_side
        new_w = int(orig_w * max_side / orig_h)
    scale = new_w*1.0 / orig_w
    canvas.width = new_w
    canvas.height = new_h


    image = original_image.resize((new_w, new_h), Image.LANCZOS)
    # Draw image on canvas
    canvas.put_image_data(np.array(image))

    if not force and os.path.exists(save_path):
        return canvas

    w = (orig_w//16)*16
    h = (orig_h//16)*16
    im = original_image.crop((0, 0, w, h))
    im.save(save_path)
    processed_image = im

    return canvas