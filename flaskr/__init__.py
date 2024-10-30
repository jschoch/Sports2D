import os

from flask import Flask, request,jsonify
from Sports2D.p2 import process_fun, setup_pose_tracker
from Sports2D.Sports2D import prep_process, DEFAULT_CONFIG2
from pathlib import Path

tracking_rtmlib = True
det_frequency = 1
mode = 'balanced'
pose_tracker = setup_pose_tracker(det_frequency, mode, tracking_rtmlib)

def get_pt():
    global pose_tracker
    if pose_tracker == None:
        print( "shit")
    else:
        print(f"found pt, not None {pose_tracker}")
    return pose_tracker

def create_app(test_config=None):
    global pose_tracker
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    #app.config.from_mapping(
        #SECRET_KEY='dev',
        #DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    #)

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

    @app.route("/gettrc")
    def trc():
        pose_tracker = get_pt()
        vidpath = request.args.get("path")
        if vidpath != None:
            vidp = Path(vidpath)
            if os.path.exists(vidp):
                trc_data = process_fun(config_dict, vidp, time_range, frame_rate, result_dir,pose_tracker)
                message = request.args.get('message')
                print(f" the message: {message}")
                #return jsonify(trc_data)
                #return trc_data.to_json()
                return jsonify(trc_data['t'].to_dict())
            else:
                return "Bad file path"
        else:
            return "ERROR"

    return app