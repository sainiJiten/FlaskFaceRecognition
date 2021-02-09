import numpy as np
import face_recognition

enc = []

picture_of_me = face_recognition.load_image_file("xxx.png")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

enc.append(my_face_encoding)

unknown_picture = face_recognition.load_image_file("yyyy.jpeg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

enc.append(unknown_face_encoding)

unknown_picture1 = face_recognition.load_image_file("zzzz.jpeg")
unknown_face_encoding1 = face_recognition.face_encodings(unknown_picture1)[0]

enc.append(unknown_face_encoding1)

np.save("newenc.npy",enc)


"""

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of elon!")
else:
    print("It's not a picture of elon!")
"""