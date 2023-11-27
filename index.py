from flask import Flask, render_template, jsonify
import face_recognition
import numpy as np
import csv
from datetime import datetime
import cv2
import os
import threading
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index1.html')



def facial_recognition():
   
    try:
        video_capture = cv2.VideoCapture(0)
        # Check if the video capture is opened
        if not video_capture.isOpened():
            print("Could not open video capture")
            return
        # loading face
        student1_image = face_recognition.load_image_file("faces/student1.jpg")
        # encoding the image
        student1_encoding = face_recognition.face_encodings(student1_image)[0]
        student2_image = face_recognition.load_image_file("faces/student2.jpg")
        student2_encoding = face_recognition.face_encodings(student2_image)[0]
        # Storing names of the encodings
        known_face_encoding = [student1_encoding, student2_encoding]
        known_face_names = ["Student1", "Student2"]
        # list of students
        students = known_face_names.copy()

        face_locations = []
        face_encodings = []

        # getting the current date and time
        now = datetime.now()
        current_date = datetime.now().strftime('%Y-%m-%d')
        f = open("attendance.csv", "a+", newline="")
        lnwriter = csv.writer(f)

        while True:
            _, frame = video_capture.read()
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            cv2.imshow('small_frame', frame)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('rgb_small_frame', frame)
            # Recognizing faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    if name not in students:
                        name = "unknown student"
                        current_time = now.strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time, current_date])
                    if name in students:
                        students.remove(name)
                        current_time = now.strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time, current_date])

                cv2.imshow("Attendance ", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
            f.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        return

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    # Start the facial recognition process in a new thread
    threading.Thread(target=facial_recognition).start()
    return jsonify({"message": "Facial recognition started."})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    filename = "attendance.csv"
    if not os.path.isfile(filename):
        return jsonify({"message": "Attendance data for today has not been created yet."})

    attendance = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            attendance.append({"name": row[0], "time": row[1], "date":row[2]})
    return jsonify(attendance)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
