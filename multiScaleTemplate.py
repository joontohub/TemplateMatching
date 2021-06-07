import numpy as np
import argparse
import imutils
import glob
import cv2
from PIL import Image


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True,
                help="Path to template image")
ap.add_argument("-i", "--images", required=True,
                help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
                help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template--", template)

image = cv2.imread("./woman_original.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None


for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])

    if resized.shape[0] < tH or resized.shape[1] < tW:
        # 여기서 제일 잘 맞는 거 찾아짐 .
        break

    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    if args.get("visualize", False):

        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(
            clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        cv2.imshow("Visualize---", clone)
        cv2.waitKey(0)

    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)


(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))


####

cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 255), 1)
print(startX, "--- startX value ---", startY, "----startY--- ",
      endX, "--endX value-- ", endY, "--endY value--")
cv2.imshow("Image---", image)
cv2.imwrite("woman_rectangled.jpg", image)
cv2.waitKey(0)


img = Image.open('./woman_synthesis.png')
newX = endX - startX
newY = endY - startY
img_resize = img.resize((newX, newY))

###

img_resize.save('./woman_resized.jpg')

# img_resize_lanczos = img.resize((256, 256), Image.LANCZOS)
# img_resize_lanczos.save('data/dst/sample_pillow_resize_lanczos.jpg')


original = Image.open('woman_original.jpg')
resized = Image.open('woman_resized.jpg')


new_image = original
new_image.paste(resized, (startX, startY, endX, endY))
new_image.save("woman_paste_result.jpg")
