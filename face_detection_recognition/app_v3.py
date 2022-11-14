''' import section '''
import os
from importlib import import_module
from unicodedata import name
from flask import Flask, render_template, Response

# timestamp and location data from influxdb
from data_query import send_puredata, send_dictdata

# class User():
#     y = []
#     z = send_dictdata(y, data_range = '10d')
#     name = []
#     location = []
#     timestamp = []
#     for i in range(len(z)):
#         name = z[i]['name']
#         location = z[i]['locate']
#         timestamp = z[i]['timestamp']

# routing paths
app = Flask(__name__)

@app.route('/')
def index():
    """ Home page """
    return render_template('index_test1.html')

@app.route('/list')
def listdata():
    """ data quering from Influxdb """
    headings = ['Name', 'Location', 'Date', 'Time']
    pure_data = []
    return render_template('data.html', headings=headings, data=send_puredata(pure_data, data_range = '30d'))

@app.route('/list2')
def listdata2():
    headings = ['Name', 'Location', 'Timestamp']
    y = []
    return render_template('data2.html', headings=headings, data=send_puredata(y, data_range = '24h'))

# import camera driver (I duplicate base_camera.py for each camera in use)
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
    Camera2 = import_module('camera_' + os.environ['CAMERA']).Camera
    Camera3 = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera, Camera2, Camera3

# Video streaming generator function.
def gen(camera):
    """ generate frame from cameras function """
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

# Video streaming route. Put this in the src attribute of an img tag.
@app.route('/video_feed')
def video_feed():
    """ 1st camera recording room """
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    """ 2nd camera IoT room """
    return Response(gen(Camera2()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    """ 3rd camera back stair """
    return Response(gen(Camera3()), mimetype='multipart/x-mixed-replace; boundary=frame')

# run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
