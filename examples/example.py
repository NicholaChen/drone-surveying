import drone_surveying
from pupil_apriltags import Detector

d = drone_surveying.DroneData(
    flight_csv_file="Flight-Airdata.csv",
    video_files=["DJI.MP4"],
    fps=100,
    max_missing_frames=1000,
    timestep=0.001,
    AprilTagDetector= Detector(
        families="tag36h11",
        nthreads=12,
        debug=0
    ),
    debug=True
)