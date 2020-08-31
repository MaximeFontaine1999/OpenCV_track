from collections import deque
from imutils.video import VideoStream
import cv2
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())


# Détermine l'intervalle de la couleur souhaité en HSV
greenLower = (30, 70, 30)
greenUpper = (94, 255, 255)

redLower = (112, 76, 86)
redUpper = (189, 255, 218)

pts = deque(maxlen=64)

# Si la video n'est pas supportée ,on se réfère à la webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# Sinon on prend la video
else:
    vs = cv2.VideoCapture(args["video"])


# Determine la forme

def detectshape(c):
    peri = cv2.arcLength(c, True)
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)
    sides = len(vertices)
    shape : str = 'unknown'
    if sides == 3:
        shape = 'triangle'
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(c)
        aspectratio = float(w) / h
        if aspectratio == 1:
            shape = 'carré'
        else:
            shape = "rectangle"
    elif sides == 5:
        shape = 'pentagone'
    elif sides == 6:
        shape = 'hexagone'
    elif sides == 8:
        shape = 'octogone'
    elif sides == 10:
        shape = 'étoile'
    else:
        shape = 'cercle'
    return shape

def print_rectangle(img, shape, c):
    if shape == 'rectangle':
        moment = cv2.moments(c)
        cx = 0
        cy = 0
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
        cv2.drawContours(img, [c], -1, (125, 0, 0), 2)
        cv2.putText(img, 'rectangle vert', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 125), 2)


while True:
    cnts=[]
    img = vs.read()

    # handle the frame from VideoCapture or VideoStream
    img = img[1] if args.get("video", False) else img

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if img is None:
        break

    img = imutils.resize(img, width=600)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Construction des masques vert et rouge
    mask1 = cv2.inRange(hsv, greenLower, greenUpper)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)

    mask2 = cv2.inRange(hsv, redLower, redUpper)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    cntg = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntg = imutils.grab_contours(cntg)

    cntr = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntr = imutils.grab_contours(cntr)

    # Transformation de l'image pour obtenir les formes observées
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
    edged = cv2.Canny(gray, 70, 150)

    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # on croise les contours des différents masque (vert et rouges) avec ceux qui ont une forme détectable
    for cg in contours and cntg:
        cnts.append(cg)
        shape = detectshape(cg)
        print_rectangle(img, shape, cg)

    for cr in contours and cntr:

        shape = detectshape(cr)
        moment = cv2.moments(cr)
        cx = 0
        cy = 0
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
        cv2.drawContours(img, [cr], -1, (0, 0, 200), 2)
        cv2.putText(img, shape + ' rouge', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 125), 2)

## si l'on souhaite ne récupérer que le plus grand objet détecté
    # if len(cnts) != 0:
        # c = max(cnts, key=cv2.contourArea)
        # shape = detectshape(c)
        # print_rectangle(img, shape, c)

    cv2.imshow("img", img)
    key = cv2.waitKey(1) & 0xFF

    # dès que l'on presse la lettre q le flux s'arrete
    if key == ord("q"):
        vs.stop()
        break
