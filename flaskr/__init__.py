import os

from flask import Flask, request,jsonify,make_response
from Sports2D.p2 import process_fun, setup_pose_tracker
from Sports2D.Sports2D import prep_process, DEFAULT_CONFIG2
from pathlib import Path
import math

import pandas as pd
import socketio
from flaskr.intern import run_inference, rewrite_file_path
import json
from flaskr.util import euclidean_distance, butter_lowpass, butter_lowpass_filter

#sio = socketio.Server(async_mode='gevent')
#sio = socketio.Client(logger=True, engineio_logger=True)
sio = socketio.Client(logger=True)


tracking_rtmlib = True
det_frequency = 1
#mode = 'balanced'
mode = 'lightweight'
pose_tracker = setup_pose_tracker(det_frequency, mode, tracking_rtmlib)




def gen_speed(data,apply_filters=True, fs=120, cutoff=12):
    calculate_constant = False
    if calculate_constant:
        a = (214.4791,974.7798)
        b = (655.811,973.405)
        ab_dist = euclidean_distance(a,b)
        print(f"the distance was: {ab_dist}")
        pixel_to_inches = 46 / ab_dist
        pixels_to_meters = 1.1684 / ab_dist
        print(f"the pixels per inch factor is: {pixel_to_inches}")
        print(f"m/p: {pixels_to_meters}")

    pixels_per_inch = 0.10422944004466449
    mps_const = 0.0026474277771344782

    #  this seesm stupid, why not fix the apply below?
    #print(f"columns: {data_in.columns}") 
    #data = pd.DataFrame(data_in)
    # fixor X and Y coordinates
    #data['X1'] = data.apply(lambda x: x[0] *1000)
    #data['Y1'] = data.apply(lambda x: x[1] * -1000)
    #data['X1'] = data.apply(lambda x: x['x'] *1000)
    #data['Y1'] = data.apply(lambda x: x['y'] * -1000)

    data['X1'] = data.x * 1000
    data['Y1'] = data.y * -1000

    
     # Initialize Distance and Speed for the first row
    data.at[0, 'Distance'] = None  # or some default value if applicable
    data.at[0, 'Speed'] = None   # or some default value if applicable

    data['Distance'] = data.apply(lambda row: euclidean_distance((row['X1'], row['Y1']),
        (data.iloc[row.name-1]['X1'], data.iloc[row.name-1]['Y1']) 
        if row.name > 0 else None), axis=1)

    # Calculate speed in units per second (assuming distance is in units and time is in seconds)
    fps_120 = 1.0/120
    #convert meters per second to miles per hour
    mps_mph_const = 2.237
    
    data['Speed'] = data['Distance'].apply(
        #lambda x: x / (frame_time_ms * mps_const * 1e-6) 
        lambda x: (x / fps_120) * mps_const * mps_mph_const
        if x is not None else 0)  # Assuming constant time_per_frame

    # Calculate time for each frame
    #data['Time'] = data.index * frame_time_ms

    if apply_filters:
        data['Distance'] = data['Distance'].interpolate(method='linear', limit_direction='both')
        data['Speed'] = data['Speed'].interpolate(method='linear', limit_direction='both')

        data['Distance_filtered'] = butter_lowpass_filter(data['Distance'], cutoff, fs)
        data['Speed_filtered'] = butter_lowpass_filter(data['Speed'], cutoff, fs)
        
    

    return data
    

def get_pt():
    global pose_tracker
    if pose_tracker == None:
        print( "shit")
    else:
        print(f"found pt, not None {pose_tracker}")
    return pose_tracker

def create_app(test_config=None):
    """ 
    i guess thsi is like an init for a class...
     
    """
    global pose_tracker
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    @sio.event
    def connect():
        print('connected to server')

    @sio.event
    def do_ocr(data_txt):
        print(f"RAW do_ocr request\n{data_txt}")
        data = json.loads(data_txt)
        #print('OCR request received for file: {0}'.format(data['file_path']) 
        ocr_data_text = run_inference(data['file_path'])
        response_data = {'ocr_data_text': ocr_data_text,'swingid': data['swingid']}
        response_text = json.dumps(response_data)
        sio.emit("ocr_data",response_text)
        print("sent done")

    @sio.event
    def do_vid(request_text):
        """
        handles trc video processing 
        """
        request_data = json.loads(request_text)
        file_path = request_data['file_path']
        swingid = request_data['swingid']
        vtype = request_data['vtype']
        rewritten_path = rewrite_file_path(file_path )

        print('Video request received for file: {0}'.format(rewritten_path))
        txt = "ERROR"
        try:
            txt = get_trc(rewritten_path,swingid,vtype)
        except Exception as e:
            print(f"Error {e}")

        sio.emit('video_data', txt)

    #uri = "http://192.168.1.220:5004/remote"
    uri = "http://192.168.1.216:5004/remote"

    print("trying to connect")
    if not sio.connected:
        sio.connect(uri)
    

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    
    print("this is init i guess")
    
    config_dict,video_file, time_range, frame_rate, result_dir = prep_process(DEFAULT_CONFIG2)
    #trc_data = process_fun(config_dict, video_file, time_range, frame_rate, result_dir)


    #TODO: get rid of this and gen_speed
    #def pre_speed(lw,key):
       #lw = lw[key]
       #lw.columns = ["x","y","z"] 
       #lw = gen_speed(lw)
       #print(f"{key}: {lw.head()}")
       #return lw

    @app.route("/gettrc")
    def trc():
        pose_tracker = get_pt()
        vidpath = request.args.get("path")
        response = get_trc2(vidpath)
        return response
        
    def get_trc2(vidpath):
        return "ERROR, not done"
        
    def get_trc(vidpath,swingid,vtype):
        #response = make_response()
        # TODO: fix this, it no longer works with the flask http request
        if vidpath != None:
            vidp = Path(vidpath)
            if os.path.exists(vidp):
                trc_data = process_fun(config_dict, vidp, time_range, frame_rate, result_dir,pose_tracker)
                #print(f"trc data head\n{trc_data.head()}")
                print("saving json")
                response_data = {
                    "trc_txt": trc_data.to_csv(),
                    "vtype":vtype,
                    "swingid":swingid
                }
                txt = json.dumps(response_data)
                print(f"returning txt {txt[:200]}")
                return txt
            else:
                s = f"The file path was bad: {vidpath}"
                #response.set_data("This is an example response")
                #response.set_status(500)
                #return response
        else:
            s = f"The file path was None!: {vidpath}"
            #response.set_data("This is an example response")
            #response.set_status(500)
            #return response

    return app