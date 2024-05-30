import cv2
import numpy as np

def overlay_sunglasses(face_img, sunglasses_img, face_cascade):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        sunglasses_width = w
        sunglasses_height = int(sunglasses_img.shape[0] * (sunglasses_width / sunglasses_img.shape[1]))

        x_offset = x
        y_offset = y + int(h / 4)

        resized_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)

        roi = face_img[y_offset:y_offset + sunglasses_height, x_offset:x_offset + sunglasses_width]

        mask = resized_sunglasses[:, :, 3]
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(resized_sunglasses[:, :, :3], resized_sunglasses[:, :, :3], mask=mask)

        dst = cv2.add(img1_bg, img2_fg)

        face_img[y_offset:y_offset + sunglasses_height, x_offset:x_offset + sunglasses_width] = dst

    return face_img

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    sunglasses_img = cv2.imread('cool.png', cv2.IMREAD_UNCHANGED)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_sunglasses = overlay_sunglasses(frame, sunglasses_img, face_cascade)

        cv2.imshow('AR Sunglasses', frame_with_sunglasses)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
