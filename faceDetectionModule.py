import cv2
import mediapipe as mp
import time



class FaceDetector():
    def __init__(self, minDetectionConf = 0.5):

        self.minDetectionConf = minDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        # aumentando min_detection_confidence, evitamos detecção duvidosas:
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConf)

    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB) #detectando

        bboxes = [] #lista com cada um dos rostos detectados, contendo id, score, posição/tamanho
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id, detection)
                #mpDraw.draw_detection(img, detection) #desenhando padrão
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih) # / : continuar código na próxima linha


                bboxes.append([id, bbox, detection.score]) #adicionando a lista

                if draw:
                    #desenhando marcação de cada rosto por conta própria:
                    img = self.fancyDraw(img, bbox)
                
                    #score: precisão
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        return img, bboxes


    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1  = x + w, y + h

        cv2.rectangle(img, bbox, (225, 0, 255), rt) #bbox: igual a passar ...(x,y), (w, h)...

        #Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        #Top Right x1,y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        #Bottom Left x,y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        #Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture('videos/1.mp4')
    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()

        img, bboxes = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 180, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

# se estamos rodando este arquivo:
if __name__ == "__main__":
    main()