import numpy as np
import face_recognition as fr
import flask
import cv2


def facer(test_img):
    person = ""
    face_locations = []
    d = dict()
    d["name"] = person
    d["rect"] = face_locations
    encodings = np.load("newenc.npy")
    fl = fr.face_locations(test_img)
    if len(fl)>0:
        face_locations = fl[0]
        test_enc = fr.face_encodings(test_img)[0]
    
        for i in range(len(encodings)):
            result = fr.compare_faces([encodings[i,:].T], test_enc)
            if result[0] == True:
                if i == 0:
                    person = "Cayden P"
                    break
                elif i == 1:
                    person = "Jiten S"
                    break
                elif i == 3:
                    person = "Zhao Lu"
                    break
            else:
                person = ""
    
        d["name"] = person
        d["rect"] = face_locations

    return d

app = flask.Flask(__name__)

@app.route('/',methods=['GET','POST'])
def handle_request():
    imagefile = flask.request.files['image']
    image = np.asarray(bytearray(imagefile.read()),dtype=np.uint8)
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    dic = facer(image)
    return dic

app.run(host="0.0.0.0",port=5000,debug=True)







