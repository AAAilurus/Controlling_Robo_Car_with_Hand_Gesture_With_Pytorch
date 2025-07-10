

import cv2
import os


gesture_label = 'left'  
dataset_path = 'Hand_gesture_datasets'
save_path = os.path.join(dataset_path, gesture_label)
os.makedirs(save_path, exist_ok=True)


cap = cv2.VideoCapture(0)
img_count = 0
max_images = 200  

print(f"Collecting images for '{gesture_label}'... Press 's' to save, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{gesture_label} - Images: {img_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Webcam - Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        img_name = os.path.join(save_path, f'{gesture_label}_{img_count}.jpg')
        cv2.imwrite(img_name, roi)
        print(f"Saved {img_name}")
        img_count += 1

        if img_count >= max_images:
            print(f"Collected {max_images} images for '{gesture_label}'.")
            break

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
