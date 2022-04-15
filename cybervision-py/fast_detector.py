from machine import detect
from PIL import Image, ImageOps

def fast_points(img: Image):
    img_adjusted = ImageOps.autocontrast(img)
    points = detect(img_adjusted, 15, 12, True)
    for point in points:
        img.putpixel(point, 255)
    img.show()
