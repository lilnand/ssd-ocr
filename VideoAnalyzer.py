import cv2
import pytesseract
import numpy as np

image_output_dir = "data"
video_path = "round1.mp4"

def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def invert_image(image):
    return cv2.bitwise_not(image)

def threshold_image(image, min_value=160):
    _, binary_image = cv2.threshold(image, min_value, 255, cv2.THRESH_BINARY)
    return binary_image    

def preprocess_image(image, min_value):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted_image = cv2.bitwise_not(gray_image)
    _, binary_image = cv2.threshold(inverted_image, min_value, 255, cv2.THRESH_BINARY)

    return binary_image

# Function to perform OCR on an image
def ocr_image(image):
    try:
        custom_config = r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(image, lang='ssd', config=custom_config)
        return text.strip()
    except Exception as e:
        return str(e)

def print_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")

if __name__ == "__main__":
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', print_coordinates)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    count = 0

    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        if count % (int(fps)) == 0:
            cv2.imshow('Video', frame)

            # crop frames: y_strat:y_end, x_start:x_end
            temperature_frame = frame[250:293, 487:580]
            voltage_frame = frame[172:292, 135:314]

            # process frames
            blackwhite_temperature = preprocess_image(temperature_frame, 182)
            blackwhite_voltage = threshold_image(voltage_frame, 99)

            print(ocr_image(blackwhite_temperature))
            print(ocr_image(blackwhite_voltage))

            cv2.imshow('temperature', blackwhite_temperature)
            cv2.imshow('voltage', blackwhite_voltage)

            key = cv2.waitKey(0)
    

    video.release()
