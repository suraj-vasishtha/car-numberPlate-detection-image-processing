import cv2
import os
import winsound

# Path to the Haar Cascade XML file
harcascade = "model/haarcascade_russian_plate_number.xml"

# Create output directory if it doesn't exist
output_dir = "plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video capture from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Minimum area of the detected number plate
min_area = 500
count = 0

# Load the plate cascade classifier
plate_cascade = cv2.CascadeClassifier(harcascade)
if plate_cascade.empty():
    raise IOError("Error loading Haar cascade xml file.")

if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        
        plate_detected = False

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                plate_detected = True
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                img_roi = img[y: y + h, x: x + w]
                cv2.imshow("ROI", img_roi)

        cv2.imshow("Result", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if plate_detected:
                if 'img_roi' in locals():
                    # Generate a unique filename
                    filename = os.path.join(output_dir, "scanned_img_" + str(count) + ".jpg")
                    # Save the ROI image permanently
                    cv2.imwrite(filename, img_roi)
                    # Display a message on the screen indicating the image is saved
                    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                    cv2.imshow("Result", img)
                    # Make a beep sound (only works on Windows)
                    if os.name == 'nt':
                        winsound.Beep(1000, 200)  # Beep at 1000 Hz for 200 ms
                    cv2.waitKey(500)  # Wait for 500 milliseconds to show the message
                    count += 1
                else:
                    print("No ROI image to save.")
            else:
                print("No plate detected to save.")
        elif key == ord('q'):
            print("Quitting the program.")
            break

cap.release()
cv2.destroyAllWindows()