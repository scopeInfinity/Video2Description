import re

def caption_tokenize(caption):
    caption = re.sub('[^a-zA-Z]+', ' ', caption).lower()
    caption = caption.split()
    return caption
