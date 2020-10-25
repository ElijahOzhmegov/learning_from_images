import cv2
from datetime import datetime

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


def timeit(method):
    def timed(*args, **kw):
        before = datetime.now()
        result = method(*args, **kw)
        after = datetime.now()
        diff = 1e6/(after - before).microseconds
        add_note_on_the_picture(result, "FPS " + str(round(diff, 2)), label_center=(0, 0))

        return result

    return timed


def wavelength_to_rgb(wavelength, gamma=0.8):
    '''
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return int(B), int(G), int(R)


def k_to_rgb(k: int, K: int):
    lower_limit = 380
    upper_limit = 750
    wavelength = lower_limit + (upper_limit - lower_limit)*(k/K)
    return wavelength_to_rgb(wavelength)

