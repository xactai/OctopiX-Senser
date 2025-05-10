import os
import cv2


def get_number_of_classes():
    with open(os.path.join(os.getcwd(), 'assets', 'imagenet_classes.txt'), 'r') as classes_file:
        classes = classes_file.readlines()
    return classes

def calculate_text_scale(image_width, image_height, text_width, text_height):
    if text_width < 0.5*image_width:
        scale = 1
    elif text_width < 0.75*image_width:
        scale = 0.75
    else:
        scale = 0.5
    return scale

def display_result(image, class_label, confidence):
    image_width, image_height = image.shape[:2]
    text = f'{class_label} : {confidence:.4f}'
    (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    scale = calculate_text_scale(image_width, image_height, w, h)
    # cv2.rectangle(image, (50,50), (50+w, 50+h*2), (118,218,244), -1)
    cv2.rectangle(image, (50,50), (50+w, 50+h*2), (0, 0, 0), -1)
    cv2.putText(image, text, (50, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2)
    return image
