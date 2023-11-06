import face_recognition
import cv2
import os

# Path to the directory containing face images
known_faces_dir = "D:\\home\\tjamil\\NED_Pgd\\PGD-DL-B3n4\\09-10\\6.2 Project\\FaceBank"

# Load known faces and their names from the directory
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(known_faces_dir):
    if file_name.endswith(".jpg"):
        
        name = os.path.splitext(file_name)[0]
        image_path = os.path.join(known_faces_dir, file_name)
        
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "No Match"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle and label the face in the frame
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()

