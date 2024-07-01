from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import getCar,read_license_plate, write_csv
mot_tracker = Sort()
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./models/license_pd.pt")
results = {}
cap = cv2.VideoCapture("./assets/traffic_video.mp4")
ret = True
frame_num = -1
vehicles = [2, 3, 5 , 6 , 7]
while ret:
    frame_num += 1
    ret, frame = cap.read()
    if ret :
        results[frame_num]= {}
        # Detecting the vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1 , x2, y2 , conf_score , class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1 , x2, y2 , conf_score])

        # Tracking the vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        #detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1 , x2, y2 , conf_score , class_id = license_plate
            # assign license plate to the car, and return is coordinates of car these license plates belong to
            xcar1, ycar1 , xcar2 , ycar2 , car_id = getCar(license_plate=license_plate, vehicle_track_ids=track_ids)
            if car_id !=-1:

            # Cropping the license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2),:]

                #Processing the license_plate, Image processing filters, gray scale conversion
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _ ,license_plate_crop_threshold =  cv2.threshold(license_plate_crop_gray, 64 , 255 , cv2.THRESH_BINARY_INV)

                ## This is the image we will convert it into the OCR technology(Easy OCR)
                license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_crop_threshold)

                if license_plate_text is not None: 
                    results[frame_num][car_id] = {
                        "car":{'bbox':[xcar1, ycar1 , xcar2 , ycar2]}, 
                        "license_plate":{'bbox':[x1, y1 , x2, y2], 'text':license_plate_text , 'bbox_score':conf_score, 'text_score':license_plate_text_conf_score}
                    }
                    

write_csv(results, "./test.csv")