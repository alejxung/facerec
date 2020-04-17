import face_recognition
import os
import cv2
import pickle
import time

KNOWN_FACES_DIR = "known_faces_id"
# UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # "cnn" or "hog" (recommended)

video = cv2.VideoCapture("facerecvid.mp4")  # cv2.VideoCapture(0) for FaceCam

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    if name.endswith(".DS_Store"):
        continue
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        # if filename.endswith(".DS_Store"):continue
        # image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        # [fn for fn in name if not fn.endswith(".DS_Store")]
        # encoding = face_recognition.face_encodings(image)[0]
        encoding = pickle.load(open(f"{name}/{filename}", "rb"))
        known_faces.append(encoding)
        known_names.append(int(name))

if len(known_faces) > 0:
    next_id = max(known_names) + 1
else:
    next_id = 0

print("processing unknown faces")

while True:
    # print(filename)
    # if filename.endswith(".DS_Store"):continue
    # image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    ret, image = video.read()
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
            pickle.dump(
                face_encoding,
                open(f"{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl", "wb"),
            )

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = [0, 255, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(
            image,
            match,
            (face_location[3] + 10, face_location[2] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            FONT_THICKNESS,
        )

    cv2.imshow("", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # cv2.waitKey(0)
    # cv2.destroyWindow(filename)
