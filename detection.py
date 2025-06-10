from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model_path = "/content/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Initialize webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
frame_width = 320
frame_height = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Start real-time detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame to match desired resolution
    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLO prediction (no verbose output, with confidence threshold)
    results = model.predict(source=frame_resized, conf=0.4, verbose=False)

    # Draw detection boxes
    if results:
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            # Draw red box
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Detection (Press 'q' to Quit)", frame_resized)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
