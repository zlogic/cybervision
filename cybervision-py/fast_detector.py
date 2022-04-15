from fast import detect
from PIL import Image

def fast_points(img: Image):
    img_data = img.tobytes('raw','L')
    points = detect(img_data, img.width, img.height, 15, 12, True)
    for point in points:
        img.putpixel(point, 255)
    img.show()
