print(
    " \n ------------------------------ SPAAC - STUDENT PERFORMANCE & ATTENTION ANALYSIS IN CLASS ------------------------------ \n "
)


# --> IMPORTS <--
print(" GETTING IMPORTS ")
import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import datetime
import pandas as pd
import matplotlib.pyplot as plt


# --> SWITCHING WEBCAM INPUT <--
# --> IF FALSE, LOADS VIDEO FROM SOURCE FILE <--
print(" WEBCAM INPUT = TRUE ")
USE_WEBCAM = True


# --> PARAMETERS FOR LOADING IMAGES & DATASETS <--
print(" GETTING PARAMETERS FOR LOADING IMAGES & DATASETS ")
emotion_model_path = "./models/emotion_model.hdf5"
emotion_labels = get_labels("fer2013")


# --> HYPER-PARAMETERS FOR BOUNDING THE BOXES SHAPE <--
print(" GETTING HYPER-PARAMETERS FOR BOUNDING THE BOXES SHAPE ")
frame_window = 10
emotion_offsets = (20, 40)


# --> LOADING & GENERATING MODEL <--
print(" LOADING & GENERATING THE MODEL ")
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)


# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# --> GETTING INPUT MODEL SHAPES FOR INFERENCE <--
print(" GETTING INPUT MODEL SHAPES FOR INFERENCE ")
emotion_target_size = emotion_classifier.input_shape[1:3]


# --> STARTING LISTS FOR CALCULATING MODES <--
print(" STARING A LIST FOR CALCULATING TOTAL NUMBER OF MODES ")
emotion_window = []


# --> LOADING SAMPLE IMAGES & LEARN HOW TO RECOGNIZE IT <--
print(" LOADING SAMPLE IMAGES & LEARNING TO RECOGNISE THEM")

sudhanshusingh_image = face_recognition.load_image_file("images/sudhanshu singh.jpg")
sudhanshusingh_face_encoding = face_recognition.face_encodings(sudhanshusingh_image)[0]

amanjain_image = face_recognition.load_image_file("images/aman jain.jpg")
amanjain_face_encoding = face_recognition.face_encodings(amanjain_image)[0]

mananshah_image = face_recognition.load_image_file("images/manan shah.jpg")
mananshah_face_encoding = face_recognition.face_encodings(mananshah_image)[0]

shreyasingh_image = face_recognition.load_image_file("images/shreya singh.jpg")
shreyasingh_face_encoding = face_recognition.face_encodings(shreyasingh_image)[0]

bhisajisurve_image = face_recognition.load_image_file("images/bhisaji surve.jpg")
bhisajisurve_face_encoding = face_recognition.face_encodings(bhisajisurve_image)[0]

nikhilsuri_image = face_recognition.load_image_file("images/nikhil suri.jpg")
nikhilsuri_face_encoding = face_recognition.face_encodings(nikhilsuri_image)[0]

rahiljain_image = face_recognition.load_image_file("images/rahil jain.jpg")
rahiljain_face_encoding = face_recognition.face_encodings(rahiljain_image)[0]


# --> CREATE ARRAYS OF KNOWN FACE ENCODINGS & THEIR NAMES <--
print(" GETTING THE CREATED ARRAY FOR KNOWN FACES ENCODINGS & THEIR NAMES ")
known_face_encodings = [
    sudhanshusingh_face_encoding,
    amanjain_face_encoding,
    mananshah_face_encoding,
    shreyasingh_face_encoding,
    bhisajisurve_face_encoding,
    nikhilsuri_face_encoding,
    rahiljain_face_encoding,
]


known_face_names = [
    "sudhanshu singh",
    "aman jain",
    "manan shah",
    "shreya singh",
    "bhisaji surve",
    "nikhil suri",
    "rahil jain",
]


# --> INITIALIZING SOME VARIABLES <--
print(" INITIALIZING SOME VARIABLES ")
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# --> PROCESSING THE FRAMES <--
print(" PROCESSING THE FRAMES & COMPARING THE DATASETS")


def face_compare(frame, process_this_frame):

    # --> RESIZE FRAME OF VIDEO TO 1/4 SIZE FOR FASTER FACE RECOGNITION PROCESSING <--
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # --> CONVERT THE IMAGE FROM BGR COLOR (WHICH OPENCV USES) TO RGB COLOR (WHICH FACE_RECOGNITION USES) <--
    rgb_small_frame = small_frame[:, :, ::-1]

    # --> ONLY PROCESS EVERY OTHER FRAME OF VIDEO TO SAVE TIME <--
    if process_this_frame:

        # --> FIND ALL THE FACES AND FACE ENCODINGS IN THE CURRENT FRAME OF VIDEO <--
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )
        face_namess = []

        for face_encoding in face_encodings:

            # --> SEE IF THE FACE IS A MATCH FOR THE KNOWN FACE(S) <--
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "UNKNOWN"

            # --> IF A MATCH WAS FOUND IN KNOWN_FACE_ENCODINGS, JUST USE THE FIRST ONE <--
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_namess.append(name)
            process_this_frame = not process_this_frame

        return face_namess


# --> STARTING VIDEO STREAMING <--
print(" STARTING VIDEO STREAMING")
cv2.namedWindow("window_frame")
video_capture = cv2.VideoCapture(0)


# --> LOOKING FOR PRE SPECIFIED COMMANDS <--
cap = None
if USE_WEBCAM == True:
    cap = cv2.VideoCapture(0)  # --> Webcam source <--
    print(" USING WEBCAM ")
else:
    cap = cv2.VideoCapture("./test/testvdo.mp4")  # --> Video file source <--
    print(" USING SOURCE FILE ")


while cap.isOpened():  # --> --> True: <-- <--
    ret, frame = cap.read()

    # frame = video_capture.read()[1]

    # --> GETTING FACIAL LANDMARKS & PRINTING THE FACIAL LANDMARKS <--
    landmrk = face_recognition.face_landmarks(frame)
    for l in landmrk:
        for key, val in l.items():
            for x, y in val:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # --> CONVERTING FRAMES TO REQUIRED COLOR <--
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image)

    # --> GETTING FACE LOCATIONS <--
    face_locations = face_recognition.face_locations(rgb_image)
    # print (reversed(face_locations))

    # --> GETTING FACE NAMES <--
    face_name = face_compare(rgb_image, process_this_frame)
    for face_coordinates, fname in zip(faces, face_name):

        x1, x2, y1, y2 = apply_offsets(
            face_utils.rect_to_bb(face_coordinates), emotion_offsets
        )
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except Exception as e:
            print(" An exception occurred ")
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        # --> GETTING FACE EMOTIONS <--
        emotion_prediction = emotion_classifier.predict(gray_face)
        # --> GETTING FACE EMOTIONS PROBABILITY <--
        emotion_probability = np.max(emotion_prediction)
        # --> GETTING FACE EMOTION LABEL <--
        emotion_label_arg = np.argmax(emotion_prediction)
        # --> GETTING FACE EMOTION LABEL TEXT <--
        emotion_text = emotion_labels[emotion_label_arg]
        # --> APPENDING EMOTION TEXT TO WINDOW <--
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except Exception as e:
            print(" An exception occurred ")
            continue

        # --> COMPARING EMOTION TEXT <--
        if emotion_text == "ANGRY":
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == "DISGUST":
            color = emotion_probability * np.asarray((0, 250, 250))
        elif emotion_text == "FEAR":
            color = emotion_probability * np.asarray((0, 0, 225))
        elif emotion_text == "HAPPY":
            color = emotion_probability * np.asarray((0, 255, 0))
        elif emotion_text == "SAD":
            color = emotion_probability * np.asarray((0, 0, 0))
        elif emotion_text == "SURPRISE":
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == "NEUTRAL":
            color = emotion_probability * np.asarray((255, 0, 0))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        # --> GENERATING EMOTION TEXT TO DISPLAY AT WINDOW <--
        if fname == "UNKNOWN":
            name = "UNKNOWN" + " is " + str(emotion_text)
        else:
            name = str(fname) + " is " + str(emotion_text)

        # --> GETTING BOUNDING BOXES FOR FACES WRT COLOR INTENSITY <--
        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(
            face_utils.rect_to_bb(face_coordinates),
            rgb_image,
            name,
            color,
            0,
            -45,
            0.5,
            1,
        )

        # print(" face_locations --> ", face_locations)
        # print(" face_encodings --> ", face_encodings)
        # print(" face_names -> ", face_names)
        # print(" face_name -> ", face_name)
        # print(" emotion_prediction -> ", emotion_prediction)
        # print(" emotion_probability -> ", emotion_probability)
        # print(" emotion_label_arg -> ", emotion_label_arg)
        # print(" emotion_text -> ", emotion_text)
        # print(" emotion_mode -> ", emotion_mode)

        # --> GENERATING DICTIONARY DATA FOR CSV OUTPUT <--
        data = {
            "face_name": face_name,
            "emotion_probability": emotion_probability,
            "emotion_mode": emotion_mode,
            "Attendance": 1,
            "Time": datetime.datetime.now(),
            # 'face_locations': face_locations,
            # 'emotion_prediction': emotion_prediction,
            # 'emotion_label_arg': emotion_label_arg,
            # 'emotion_text': emotion_text
        }

        # --> GENERATING DATAFRAMES FOR MANIPULATION OF DATA AT CSV <--
        df = pd.DataFrame(
            data,
            columns=[
                "face_name",
                "emotion_probability",
                "emotion_mode",
                "Attendance",
                "Time",
            ],
        )
        df.to_csv(
            r"C:\Users\admin\PycharmProjects\SPAAC\file.csv",
            mode="a",
            index=False,
            header=False,
        )
        print(df)

        print(
            " ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ "
        )

    # --> DISPLAY THE RESULTS <--
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # --> SCALE BACK UP FACE LOCATIONS SINCE THE FRAME WE DETECTED IN WAS SCALED TO 1/4 SIZE <--
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(
            frame, (left, bottom + 36), (right, bottom), (0, 0, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.3, (255, 255, 255), 1)

    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("window_frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
