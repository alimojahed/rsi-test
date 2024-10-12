import cv2
import numpy as np
from pyzbar.pyzbar import decode
from ultralytics import YOLO
from setting import logger, parser
from cv2.typing import MatLike


args = None

def debug_image(window_name:str, image:cv2.typing.MatLike) -> None :
    if args.debug:
        logger.debug(f"Display image for {window_name}")

        cv2.imshow(window_name, image)
        cv2.waitKey(0)

        logger.debug(f"Resuming image display {window_name}")
    

def get_binary_image(image:MatLike) -> MatLike :

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    debug_image("binary image", binary)

    return binary


def filling_gaps(binary:MatLike) -> MatLike:
    kernel = np.ones((25, 25), np.uint8)

    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    dilated = cv2.dilate(closing, kernel, iterations=1)

    filled_qr = cv2.bitwise_not(dilated)

    debug_image("Dialated Image", filled_qr)

    return filled_qr

def get_largest_contour(filled_qr: MatLike, original_image:MatLike) -> MatLike:
    edges = cv2.Canny(filled_qr, 50, 150)

    debug_image("Edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_image = original_image.copy()

    if args.debug:
        for i, contour in enumerate(contours):
            cv2.drawContours(contours_image, [contour], -1, (0, 255, 0), 2)

            area = cv2.contourArea(contour)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            cv2.putText(contours_image, f"Area: {int(area)}", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        debug_image("contours with area", contours_image)
        
    contour = max(contours, key=cv2.contourArea)

    return contour


def get_four_vertecies_of_qrcode(contour:MatLike, original_image:MatLike) -> np.ndarray:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) > 4:
        x, y, w, h = cv2.boundingRect(approx)
        
        new_contour = np.array([
            [[x, y]],  # Top-left
            [[x + w, y]],  # Top-right
            [[x + w, y + h]],  # Bottom-right
            [[x, y + h]]  # Bottom-left
        ])
        
        approx = new_contour

    approx_image = original_image.copy()
    cv2.drawContours(approx_image, [approx], -1, (255, 0, 0), 2)

    debug_image("approximated polygon", approx_image)

    source_points = None
    if len(approx) == 4:

        source_points = []
        for point in approx:
            point = point[0]
            min_dist = float('inf')
            nearest_point = point

            for contour_point in contour:
                contour_point = contour_point[0]
                dist = np.linalg.norm(contour_point - point)
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = contour_point

            source_points.append(nearest_point)


        source_points_image = original_image.copy()
        for pt in source_points:
            cv2.circle(source_points_image, tuple(pt), 10, (0, 255, 0), -1)

        debug_image("source points on image", source_points_image)

        source_points = np.array(source_points, dtype='float32')

    return source_points

def warp_qrcode(original_image: MatLike, source_points: np.ndarray) -> MatLike:

    width = max(np.linalg.norm(source_points[1] - source_points[0]), np.linalg.norm(source_points[2] - source_points[3]))
    height = max(np.linalg.norm(source_points[3] - source_points[0]), np.linalg.norm(source_points[2] - source_points[1]))
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(source_points, dst_points)

    warped = cv2.warpPerspective(original_image, matrix, (int(width), int(height)))


    cv2.imshow("Warped (Squared QR Code)", warped)
    cv2.waitKey(0)

    warped_path = 'images/warped_qrcode.jpg'

    cv2.imwrite(warped_path, warped)
    logger.info(f"Save warped qrcode image to {warped_path}")

    return warped


def read_qrcode_values(qrcode_image: MatLike) -> str:
    qr_codes = decode(qrcode_image)

    height, width, _ = qrcode_image.shape

    qrcode_data_list = []

    qr_code_detected_image = qrcode_image.copy()

    for qr_code in qr_codes:

        qr_data = qr_code.data.decode('utf-8')
        qr_type = qr_code.type
        qrcode_data_list.append(qr_data)

        print(f"QR Code Type: {qr_type}")
        print(f"QR Code Data: {qr_data}")

        points = qr_code.polygon

        pts = np.array([(p.x, p.y) for p in points], dtype=np.int32)

        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)  
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)  

        cv2.polylines(qr_code_detected_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        top_left = pts[np.argmin(pts.sum(axis=1))]

        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)
        bg_color = (255, 255, 255)  

        (text_width, text_height), baseline = cv2.getTextSize(qr_data, font, font_scale, font_thickness)

        rect_start = (top_left[0], top_left[1] + text_height + baseline)
        rect_end = (top_left[0] + text_width, top_left[1])

        cv2.rectangle(qr_code_detected_image, rect_start, rect_end, bg_color, thickness=cv2.FILLED)

        cv2.putText(qr_code_detected_image, qr_data, (top_left[0], top_left[1] + text_height//2 + 2 * baseline),
                    font, font_scale, text_color, font_thickness)
        

    cv2.imshow("QR Code Detection", qr_code_detected_image)
    cv2.waitKey(0)

    return qrcode_data_list[-1]

def detect_qrcode(path:str) -> str:
    image = cv2.imread(path)
    
    binary = get_binary_image(image)

    filled_qr = filling_gaps(binary)
    
    contour = get_largest_contour(filled_qr, image)

    source_points = get_four_vertecies_of_qrcode(contour, image)
    detected_qrcode_value = None

    if source_points is not None:
        warped = warp_qrcode(image, source_points)
        
        detected_qrcode_value = read_qrcode_values(warped)
        
    
    return detected_qrcode_value


def load_and_classify_result(qrcode_data:str) -> None:
    path = f"images/{qrcode_data}.jpeg"
    image_to_detected = cv2.imread(path)
    cv2.imshow("IMAGE TO CLASSIFY", image_to_detected)
    cv2.waitKey(0)

    model = YOLO(args.model)

    results = model.predict(image_to_detected)

    for result in results:
        detected_class = result.names[result.probs.top1]
        logger.info(f"classify image as {detected_class}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = ( 255, 255)  
        thickness = 2

        image_height, image_width = image_to_detected.shape[:2]

        text_size, _ = cv2.getTextSize(detected_class, font, font_scale, thickness)

        text_x = (image_width - text_size[0]) // 2  
        text_y = image_height - 20  

        cv2.putText(image_to_detected, detected_class, (text_x, text_y), font, font_scale, color, thickness)

        cv2.imshow('Classified Image', image_to_detected)
        cv2.waitKey(0)




if __name__ == "__main__":

    try:
        logger.info("Parsing Commandline Arguments...")
        args = parser.parse_args()
        
        data = detect_qrcode(args.qrcode)

        if data is not None:
            load_and_classify_result(data)
        
        else:
            logger.error("can not detect qrcode")

    finally:
        cv2.destroyAllWindows()
    