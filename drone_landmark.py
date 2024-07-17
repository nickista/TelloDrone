import face_recognition
import cv2
import numpy as np
import time

from djitellopy import Tello

previousleft = 0
previoustop = 0

drone = Tello()
drone.connect()
drone.streamon()

previous = "unknwon"
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
# video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
Biden_image = face_recognition.load_image_file("../Image/Biden.webp")
Biden_face_encoding = face_recognition.face_encodings(Biden_image)[0]

# Load a second sample picture and learn how to recognize it.
Bill_gate_image = face_recognition.load_image_file("../Image/Bill_gate.jpg")
Bill_gate_face_encoding = face_recognition.face_encodings(Bill_gate_image)[0]

# Load a sample picture and learn how to recognize it.
Jobs_image = face_recognition.load_image_file("../Image/Jobs01.jpg")
Jobs_face_encoding = face_recognition.face_encodings(Jobs_image)[0]

# Load a sample picture and learn how to recognize it.
Target_image = face_recognition.load_image_file("../Image/Me.jpg")
Target_face_encoding = face_recognition.face_encodings(Target_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Biden_face_encoding,
    Bill_gate_face_encoding,
    Jobs_face_encoding,
    Target_face_encoding
]
known_face_names = [
    "Joe Biden",
    "Bill Gate",
    "Steve Jobs",
    "Target"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    frame = drone.get_frame_read().frame
    # Grab a single frame of video
    # ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] > 0.5:
                name = known_face_names[best_match_index]

            face_names.append(name)
            # drone.takeoff()

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # if (previousleft != left or previousleft != top):
        #     print(left)
        #     print("top=")
        #     print(top)
        #     if (left > previousleft):
        #         print("moveleft")
        #         print("moveleft")
        #         # drone.left(20)
        #         previousleft = left
        #     if (top > previoustop):
        #         # drone.forward(20)
        #         print("moveup")
        #         print("moveleft")
        #         previoustop = top
        #     if (top < previoustop):
        #         # drone.backward(20)
        #         print("Movedown")
        #         print("moveleft")
        #         previousltop = top
        #     if (left < previousleft):
        #         # drone.right(20)
        #         print("right")
        #         print("moveleft")
        #         previousleft = left

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 128, 0), 2)
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 128, 0), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # previousleft = left

        # Face Landmark
        # Draw a box around the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        #
        results = face_recognition.face_landmarks(frame)
        print('results', results)

        if len(results) != 0:

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 128, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 128, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # chin = results[0]['chin']
            # right_eyebrow = results[0]['right_eyebrow']
            # left_eyebrew = results[0]['left_eyebrow']
            # right_eye = results[0]['right_eye']
            # left_eye = results[0]['left_eye']
            # nose_tip = results[0]['nose_tip']
            # bottom_lip = results[0]['bottom_lip']
            #
            # for chins in chin:
            #     # Draw a box around the face
            #     cv2.circle(frame, chins, 2, (0, 255, 0), 1)
            #
            # for r_eb in right_eyebrow:
            #     cv2.circle(frame, r_eb, 2, (0, 255, 0), 1)
            #
            # for l_eb in left_eyebrew:
            #     cv2.circle(frame, l_eb, 2, (0, 255, 0), 1)
            #
            # for l_eb in left_eyebrew:
            #     cv2.circle(frame, l_eb, 2, (0, 255, 0), 1)
            #
            # for r_e in right_eye:
            #     cv2.circle(frame, r_e, 2, (0, 255, 0), 1)
            #
            # for l_e in left_eye:
            #     cv2.circle(frame, l_e, 2, (0, 255, 0), 1)
            #
            # for l_e in nose_tip:
            #     cv2.circle(frame, l_e, 2, (0, 255, 0), 1)
            #
            # for b_l in bottom_lip:
            #     cv2.circle(frame, b_l, 2, (0, 255, 0), 1)

        # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
# video_capture.release()
# drone.land()
# bus.close()

cv2.destroyAllWindows()
