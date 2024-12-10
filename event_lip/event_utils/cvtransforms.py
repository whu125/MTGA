import random
import numpy as np

def CenterCrop(event_frame, size):
    w, h = event_frame.shape[-1], event_frame.shape[-2]
    th, tw = size
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)
    event_frame = event_frame[..., y1: y1 + th, x1: x1 + tw]
    
    
    return event_frame

def RandomCrop(event_frame, size):
    w, h = event_frame.shape[-1], event_frame.shape[-2]
    th, tw = size
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    event_frame = event_frame[..., y1: y1 + th, x1: x1 + tw]
    
    
    return event_frame

def HorizontalFlip(event_frame):
    if random.random() > 0.5:
        event_frame = np.ascontiguousarray(event_frame[..., ::-1])
       
    return event_frame
