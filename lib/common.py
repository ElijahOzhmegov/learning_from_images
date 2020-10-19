import cv2

Q_KEY     = 113
SPACE_KEY = 32

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
LINE_TYPE  = 1


def add_note_on_the_picture(img, text="Press SPACE to continue", label_center=None):
    (label_width, label_height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, LINE_TYPE+2)

    if label_center is None:
        ls = img.shape
        x = (ls[1] - label_width)//2
        y = label_height
    else:
        x = label_center[0]
        y = label_center[1]

    cv2.putText(img, text,
                (x, y),
                FONT, FONT_SCALE,
                0, LINE_TYPE+2)
    cv2.putText(img, text,
                (x, y),
                FONT, FONT_SCALE,
                FONT_COLOR, LINE_TYPE)
