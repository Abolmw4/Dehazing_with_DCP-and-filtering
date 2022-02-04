from Dehazing import Haze_remove
import cv2
import os

image_path=os.path.join(os.getcwd(),'screenshot.jpg')
image=cv2.imread(image_path)
h=Haze_remove(image,blockSize=15,meanMode=True,refine=True)
asdf=h.hazeFree(image)
cv2.imshow('hazyImage',image)
cv2.imshow('hazefreeImage',asdf)
cv2.waitKey()
cv2.destroyAllWindows()