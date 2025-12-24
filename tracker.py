import cv2
import time

import mediapipe as mp

# YOLOは環境で入ってないことが多いので、失敗したら無効化して続行する
try:
    import torch
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class App:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2
        )

        # クリック用の仮想ボタン
        self.btn = dict(x=50, y=50, w=120, h=55)
        self.last_shot = 0.0

        # YOLO
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
                self.yolo.conf = 0.35
            except Exception as e:
                print("[WARN] YOLO load failed -> object detection OFF:", e)
                self.yolo = None

    @staticmethod
    def _to_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def draw_button(self, frame):
        x, y, w, h = self.btn["x"], self.btn["y"], self.btn["w"], self.btn["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 200, 20), -1)
        cv2.putText(frame, "SHOT", (x + 18, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def in_button(self, px, py):
        x, y, w, h = self.btn["x"], self.btn["y"], self.btn["w"], self.btn["h"]
        return (x <= px <= x + w) and (y <= py <= y + h)

    def maybe_shot(self, raw_frame):
        now = time.time()
        if now - self.last_shot >= 1.0:
            self.last_shot = now
            filename = f"photo_{int(now)}.jpg"
            cv2.imwrite(filename, raw_frame)
            print("saved:", filename)

    def detect_holistic(self, frame):
        res = self.holistic.process(self._to_rgb(frame))

        if res.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, res.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        if res.face_landmarks:
            self.mp_draw.draw_landmarks(
                frame, res.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION
            )
        if res.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, res.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        if res.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, res.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )

        return frame

    def detect_objects(self, frame):
        if self.yolo is None:
            return frame

        results = self.yolo(frame)  # BGR
        df = results.pandas().xyxy[0]

        for _, r in df.iterrows():
            x1, y1, x2, y2 = int(r.xmin), int(r.ymin), int(r.xmax), int(r.ymax)
            label = str(r.name)
            conf = float(r.confidence)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        return frame

    def detect_hand_button(self, frame, raw_frame):
        res = self.hands.process(self._to_rgb(frame))
        self.draw_button(frame)

        if not res.multi_hand_landmarks:
            return frame

        for hand in res.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            tip = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            px = int(tip.x * frame.shape[1])
            py = int(tip.y * frame.shape[0])

            # 指先に赤い円を描画
            cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)

            if self.in_button(px, py):
                cv2.putText(frame, "CLICK!", (self.btn["x"], self.btn["y"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.maybe_shot(raw_frame)

        return frame


def main():
    app = App()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("camera open failed")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw = frame.copy()

        app.detect_holistic(frame)
        app.detect_objects(frame)
        app.detect_hand_button(frame, raw)

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
