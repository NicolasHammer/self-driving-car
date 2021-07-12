import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W, A, S, D

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    processed_img = roi(processed_img, [vertices])

    # Find edges of street
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
    draw_lines(processed_img, lines)
    return processed_img

def main():
    for i in reversed(range(0, 4)):
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    while True:
        # 800x600 windowed mode 
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))

        # print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        new_screen = process_img(printscreen)
        cv2.imshow("window", new_screen)
        # cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()