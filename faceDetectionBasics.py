import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/1.mp4')
pTime = 0 

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# objeto:
faceDetection = mpFaceDetection.FaceDetection(0.7) #0.75: aumentando min_detection_confidence, evitando detecção duvidosas 

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB) #detectando

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            #mpDraw.draw_detection(img, detection) #desenhando padrão
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih) # / : continuar código na próxima linha

            #desenhando marcação por conta própria:
            cv2.rectangle(img, bbox, (225, 0, 255), 2) #bbox: igual a passar ...(x,y), (w, h)...
            print(bbox)
            #score: precisão
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 180, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)