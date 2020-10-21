import cv2


FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
LINE_TYPE  = 1


class KEYS:
    Q      = ord('q')
    SPACE  = ord(' ')
    ZERO   = ord('0')
    ONE    = ord('1')
    TWO    = ord('2')
    THREE  = ord('3')
    FOUR   = ord('4')
    FIVE   = ord('5')
    SIX    = ord('6')
    SEVEN  = ord('7')
    EIGHT  = ord('8')
    NINE   = ord('9')


def add_note_on_the_picture(img, text="Press SPACE to continue", label_center=None):
    (label_width, label_height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, LINE_TYPE+2)

    if label_center is None:
        ls = img.shape
        x = (ls[1] - label_width)//2
        y = label_height
    else:
        x = label_center[0]
        y = label_center[1] + label_height

    cv2.putText(img, text,
                (x, y),
                FONT, FONT_SCALE,
                0, LINE_TYPE+2)
    cv2.putText(img, text,
                (x, y),
                FONT, FONT_SCALE,
                FONT_COLOR, LINE_TYPE)
