from collections import deque
from imutils.video import VideoStream
import cv2
import imutils

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

greenLower = (30, 70, 30)
greenUpper = (93, 255, 255)

redLower = (173, 160, 151)
redUpper = (201, 237, 200)

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
        shape = 'octagone'
    elif sides == 10:
        shape = 'étoile'
    else:
        shape = 'cercle'
    return shape


while True:
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

    # mask1 = cv2.inRange(hsv, greenLower, greenUpper)
    # mask1 = cv2.erode(mask1, None, iterations=2)
    # mask1 = cv2.dilate(mask1, None, iterations=2)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
    edged = cv2.Canny(gray, 100, 180)

    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        moment = cv2.moments(cnt)
        shape = detectshape(cnt)
        if shape == 'rectangle':

            # Construction des masques vert et rouge
            mask1 = cv2.inRange(hsv, greenLower, greenUpper)
            mask1 = cv2.erode(mask1, None, iterations=2)
            mask1 = cv2.dilate(mask1, None, iterations=2)
            mask1 = cv2.Canny(mask1, 100, 180)

            mask2 = cv2.inRange(hsv, redLower, redUpper)
            mask2 = cv2.erode(mask2, None, iterations=2)
            mask2 = cv2.dilate(mask2, None, iterations=2)

            # contours

            cntg = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntg = imutils.grab_contours(cntg)

            cntr = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntr = imutils.grab_contours(cntr)

            for cg in cntg:
                moment = cv2.moments(cg)
                cx = 0
                cy = 0
                if moment['m00'] != 0:
                    cx = int(moment['m10'] / moment['m00'])
                    cy = int(moment['m01'] / moment['m00'])
                cv2.drawContours(img, [cg], -1, (0, 0, 0), 2)
                cv2.putText(img, shape + ' vert', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for cr in cntr:
                moment = cv2.moments(cr)
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])
                peri = cv2.arcLength(cg, True)
                vertices = cv2.approxPolyDP(cg, 0.02 * peri, True)
                sides = len(vertices)
                cv2.drawContours(img, [peri], -1, (0, 0, 0), 2)
                cv2.putText(img, 'rectangle rouge', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # affiche la video à l'écran
    # cv2.imshow("edged", edged)
    cv2.imshow("img", img)
    key = cv2.waitKey(1) & 0xFF

    # dès que l'on presse la lettre q le flux s'arrete
    if key == ord("q"):
        vs.stop()
        break
