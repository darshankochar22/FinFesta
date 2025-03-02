import cv2
import face_recognition
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

KNOWN_IMAGE_PATH = "known_image.jpg"
known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
known_encoding = face_recognition.face_encodings(known_image)[0]
known_faces = [known_encoding]

UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/authenticate/")
async def authenticate_face(file: UploadFile = File(...)):
    """Authenticates a person by comparing their uploaded image with the known image."""
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and encode the uploaded image
        test_image = face_recognition.load_image_file(file_path)
        test_encodings = face_recognition.face_encodings(test_image)

        if not test_encodings:
            return {"status": "error", "message": "No face detected in the uploaded image"}

        test_encoding = test_encodings[0]
        video_capture = cv2.VideoCapture(0)
        start_time = time.time()
        retry_printed = False

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_recognition.face_locations(rgb_frame))

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([test_encoding], face_encoding)
                face_distances = face_recognition.face_distance([test_encoding], face_encoding)
                best_match = np.argmin(face_distances)

                if matches[best_match]:  
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return {"status": "Authenticated"}

            elapsed_time = time.time() - start_time

            if elapsed_time > 5 and not retry_printed:
                print("Retrying...")
                retry_printed = True  

            if elapsed_time > 10:
                print("Not Found")
                break

        video_capture.release()
        cv2.destroyAllWindows()
        return {"status": "Not Found"}

    except Exception as e:
        return {"status": "error", "message": str(e)}