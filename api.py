import os
import urllib.request
from flask import Flask, request, redirect, jsonify,send_file
from werkzeug.utils import secure_filename
from face_blur import anonymize_face_pixelate,anonymize_face_simple
import numpy as np
import cv2
import os
import base64
from flask_cors import CORS

UPLOAD_FOLDER = '/home/runner/nahjustsomerandomshittestrepo/input'
output_folder = '/home/runner/nahjustsomerandomshittestrepo/output/'
app = Flask(__name__)
cors = CORS(app)
app.secret_key = "adjksakjdhjkas"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
prototxtPath = '/home/runner/nahjustsomerandomshittestrepo/face_detector/deploy.prototxt'
weightsPath = '/home/runner/nahjustsomerandomshittestrepo/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxtPath, weightsPath)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET'])
def home():
	return'''<div><center><h1>Hello from API </h1></center></div> '''

@app.route('/file',methods=['POST'])
def upload_file():
  try:
    f_name = request.json['filename']
    filename = "{}/{}".format(UPLOAD_FOLDER,request.json["filename"])
    image64 = request.json["image64"]
    with open(filename,"wb") as f:
      f.write(base64.b64decode(image64))
    image = cv2.imread(filename)
    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > 0.5:
        box =detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = image[startY:endY, startX:endX]
        face = anonymize_face_simple(face, factor=1.5)
        image[startY:endY, startX:endX] = face
    cv2.imwrite(os.path.join(output_folder , f_name), image)
    with open("{}/{}".format(output_folder,f_name), "rb") as image_file:
      base64img = base64.b64encode(image_file.read())
    os.remove("{}/{}".format(output_folder,f_name))
    os.remove(filename)
    response = jsonify({"status":"1","message":"converted!!!","image": str(base64img)})
    response.status_code = 201
    return response
    
  except Exception as e:
    print(e)
    response = jsonify({"status":"-1",'message': 'something went wrong'})
    response.status_code = 400
    return response

@app.route('/output/<filename>')
def display_image(filename):
	file = output_folder+ filename
	return send_file(file, mimetype='image/gif')
if __name__ == '__main__':
	app.run(host ='0.0.0.0',threaded=True,port=8080)