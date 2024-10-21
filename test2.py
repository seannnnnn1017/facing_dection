import face_recognition
import cv2
import numpy as np

# 載入已知人臉的編碼
known_image = face_recognition.load_image_file("images\me.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# 初始化相機
video_capture = cv2.VideoCapture(0)

while True:
    # 捕獲視頻的每一幀
    ret, frame = video_capture.read()

    # 將圖片從 BGR 顏色（OpenCV 使用）轉換為 RGB 顏色（face_recognition 使用）
    rgb_frame = frame[:, :, ::-1]

    # 找到所有人臉和人臉編碼
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 遍歷每個人臉
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 檢查人臉是否匹配已知人臉
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Unknown"
        if matches[0]:
            name = "Me"

        # 繪製人臉框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 在框下方繪製名字
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # 顯示結果
    cv2.imshow('Video', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放相機資源
video_capture.release()
cv2.destroyAllWindows()
