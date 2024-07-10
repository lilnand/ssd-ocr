import os
import cv2
import csv
import pytesseract
import numpy as np

video_path = "round2.mp4"

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
    
    fields = ['upper_voltage', 'middle_voltage', 'lower_voltage', 'temperature']
    dataset = []

    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        if count % (int(fps)) == 0:
            cv2.imshow('Video', frame)

            # crop frames: y_strat:y_end, x_start:x_end
            upper_multimeter = frame[70:152, 1339:1594]
            middle_multimeter = frame[353:416, 1357:1578]
            lower_multimeter = frame[585:644, 1379:1573]
            temperature_frame = frame[703:740, 1077:1165]   
            
            # process frames
            blackwhite_voltage1 = preprocess_image(upper_multimeter, 175)
            blackwhite_voltage2 = preprocess_image(middle_multimeter, 175)
            blackwhite_voltage3 = preprocess_image(lower_multimeter, 175)
            blackwhite_temperature = preprocess_image(temperature_frame, 90)

            # print(ocr_image(blackwhite_temperature))
            # print(ocr_image(blackwhite_voltage))

            dataset.append({
                'upper_voltage': ocr_image(blackwhite_voltage1),
                'middle_voltage': ocr_image(blackwhite_voltage2),
                'lower_voltage': ocr_image(blackwhite_voltage3),
                'temperature': ocr_image(blackwhite_temperature)
            })


            # cv2.imshow('Upper Multimeter', blackwhite_voltage1)
            # cv2.imshow('Middle Multimeter', blackwhite_voltage2)
            # cv2.imshow('Lower Multimeter', blackwhite_voltage3)
            # cv2.imshow('Temperature Screen', blackwhite_temperature)
            # cv2.imshow('temperature', blackwhite_temperature)
            # cv2.imshow('voltage', blackwhite_voltage)
            #key = cv2.waitKey(0)

    with open(os.path.basename(video_path) + '_dataset.csv', 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(dataset)

    video.release()
