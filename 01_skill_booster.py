import cv2
import numpy as np
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "stopsign.png"
img = cv2.imread(filepath)

COLORSHSV = {
    "red1": ([0, 80, 80], [10, 255, 255]),
    "red2": ([170, 80, 80], [179, 255, 255]),
    "orange": ([11, 80, 80], [25, 255, 255]),
    "yellow": ([26, 80, 80], [34, 255, 255]),
    "green": ([35, 80, 80], [85, 255, 255]),
    "blue": ([86, 80, 80], [130, 255, 255]),
    "purple": ([131, 80, 80], [169, 255, 255]),
    "white": ([0, 0, 200], [179, 30, 255])

}


def createcolormasks(image, title, save=False):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = {}
    
    #create color masks, currently hard coded for red, orange, yellow, green, blue, and purple
    for color, value in COLORSHSV.items():
        lower = np.array(value[0], dtype=np.uint8)
        upper = np.array(value[1], dtype=np.uint8)
        masks[color] = cv2.inRange(hsv_image, lower, upper)

        #cv2.imshow(title, image)
    #handle red separately since it wraps around the hue spectrum
    masks["red"] = cv2.bitwise_or(masks.pop("red1"), masks.pop("red2"))
    
    # save images if save is True
    if save:
        for color, mask in masks.items():
            mask_image = cv2.bitwise_and(image, image, mask=mask)
            filename = f"{color}_mask.jpg"
            cv2.imwrite(filename, mask_image)
            print(f"Mask saved as {filename}")
    '''
    for color, mask in masks.items():
            cv2.imshow(f"{color} mask", mask)
    '''
    
    return masks

def findcenter_object(image,color, mask):
    

    contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"{color}: No object found")
        return
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) > 500:  # Adjust threshold as needed
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"{color}: Object found at ({cX}, {cY})")
            cv2.drawContours(image, [main_contour], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
            cv2.putText(image, f"{color} center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            print(f"{color}: Object found but moments calculation failed")
            return
    else:
        print(f"{color}: Object found but area is too small")
        return
    


masks = createcolormasks(img, "Original Image", save=True)

for color, mask in masks.items():
        findcenter_object(img, color, mask)

cv2.imshow("Detected Objects", img)
print("Masks created and saved.")
cv2.waitKey(0)
cv2.destroyAllWindows()