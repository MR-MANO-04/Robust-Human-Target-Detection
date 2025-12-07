import cv2
import time
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO


class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectCentroidsArr = np.array(objectCentroids)
            D = np.linalg.norm(objectCentroidsArr[:, None, :] - inputCentroids[None, :, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


def main():
    model = YOLO("yolov8n.pt")
    video_path = "cctv_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    ct = CentroidTracker(max_disappeared=30, max_distance=60)
    entry_count = 0
    exit_count = 0
    prev_centroids = {}
    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = current_time
        results = model.predict(frame, conf=0.35, verbose=False)
        rects = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    rects.append((int(x1), int(y1), int(x2), int(y2)))
        objects = ct.update(rects)
        h, w = frame.shape[:2]
        line_y = h // 2
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        for (box, _) in zip(rects, range(len(rects))):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for objectID, centroid in objects.items():
            cX, cY = centroid
            if objectID in prev_centroids:
                prev_cX, prev_cY = prev_centroids[objectID]
                if prev_cY < line_y <= cY:
                    entry_count += 1
                elif prev_cY > line_y >= cY:
                    exit_count += 1
            prev_centroids[objectID] = centroid
            cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {objectID}", (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Entry: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Exit: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Human Detection and Counting - CCTV Entry/Exit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
