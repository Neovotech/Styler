import numpy as np
import cv2

def encode(img):
    # img = cv2.imread("./male-black-suit-white-shirt-upper-body-6.jpg")
    _, encoded_image = cv2.imencode('.jpg', img)

    return encoded_image.tobytes()

if __name__ == '__main__':
    byte_img = encode()

    np_img = np.asarray(bytearray(byte_img), dtype='uint8')
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    cv2.imwrite("test.jpg", img)