# written by NC

import math
import os
import cv2
from matplotlib import pyplot as plt
import pyproj
import csv
from pupil_apriltags import Detector
from scipy.interpolate import CubicSpline
from scipy.signal import correlate, correlation_lags
import numpy as np
from mpl_axes_aligner import align

_RED = '\033[31m'
_GREEN = '\033[32m'
_BLUE = '\033[34m'
_YELLOW = '\033[33m'
_CYAN = '\033[36m'
_MAGENTA = '\033[35m'
_WHITE = '\033[37m'
_BLACK = '\033[30m'
_BOLD = '\033[1m'
_ITALIC = '\033[3m'
_UNDERLINE = '\033[4m'
_RESET = '\033[0m'


class DroneData:
    def __init__(self, flight_csv_file: str, 
                 video_files: list[str], 
                 fps: float = 29.97, 
                 compass_heading: float = 0.0,
                 altitude: float = 5.0,
                 resolution: tuple[int, int] = (3840, 2160),
                 fov: float = 26.0,
                 gimbal_pitch: float = -90.0, # not used yet
                 max_missing_frames: int = 10,
                 timestep: float = 0.001,
                 AprilTagDetector: Detector = Detector(
                    families="tag36h11",
                    debug=0
                 ),
                 pyprojTransformer: pyproj.Transformer = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:3158"), always_xy=True), # WGS84 to UTM zone 14N
                 debug: bool = False
                 ):
        """
        Initializes the DroneData class with flight data from a CSV file and video files.
        :param flight_csv_file: Path to the flight CSV file.
        :type flight_csv_file: str
        :param video_files: List of paths to video files.
        :type video_files: list[str]
        :param fps: Frames per second of the video.
        :type fps: float
        :param compass_heading: Compass heading of the drone in degrees.
        :type compass_heading: float
        :param altitude: Altitude of the drone in meters.
        :type altitude: float
        :param resolution: Resolution of the video in pixels (width, height).
        :type resolution: tuple[int, int]
        :param fov: Field of view of the camera in degrees.
        :type fov: float
        :param gimbal_pitch: Gimbal pitch angle in degrees.
        :type gimbal_pitch: float
        :param max_missing_frames: Maximum number of consecutive frames that can be missing AprilTags before stopping detection.
        :type max_missing_frames: int
        :param timestep: Time step for interpolation in seconds.
        :type timestep: float
        :param AprilTagDetector: Instance of the AprilTagDetector for detecting AprilTags in the video.
        :type AprilTagDetector: Detector
        :param pyprojTransformer: Instance of the pyproj.Transformer for coordinate transformation.
        :type pyprojTransformer: pyproj.Transformer
        :param debug: If True, enables debug mode with additional output.
        :type debug: bool
        """

        self.flight_csv_file = flight_csv_file.replace("\\", "/")
        self.video_files = [video_file.replace("\\", "/") for video_file in video_files]
        self.fps = fps
        self.compass_heading = compass_heading % 360.0
        self.altitude = altitude
        self.gimbal_pitch = gimbal_pitch
        self.max_missing_frames = max_missing_frames
        self.timestep = timestep
        self.resolution = resolution
        self.fov = fov

        self.AprilTagDetector = AprilTagDetector
        self.pyprojTransformer = pyprojTransformer

        self.debug = debug

        if not os.path.exists(flight_csv_file):
            raise Exception(f"{_RED}Flight CSV file does not exist: {flight_csv_file}{_RESET}")
        flight_csv_file_reader = csv.DictReader(open(flight_csv_file, "r"))

        last = -1
        lastVideo = "0"

        timestamps = np.array([])
        x = []
        y = []

        self.video_start_estimates = [] # used to estimate the start of videos
        for row in flight_csv_file_reader:
            if int(row["time(millisecond)"]) != last:
                last = int(row["time(millisecond)"])
                timestamps = np.append(timestamps, int(row["time(millisecond)"]) / 1000)

                lat = float(row["latitude"])
                lon = float(row["longitude"])

                _x, _y = self.pyprojTransformer.transform(lon, lat)
                
                x.append(_x)
                y.append(_y)

                if row["isVideo"] == "1" and lastVideo == "0": # checks when the video starts (ie when isVideo changes from 0 to 1)
                    self.video_start_estimates.append(int(row["time(millisecond)"]) / 1000)
                lastVideo = row["isVideo"]

        x = np.array(x)
        y = np.array(y)

        self.x = CubicSpline(timestamps, x)
        self.y = CubicSpline(timestamps, y)

        print(f"{_GREEN}Flight CSV File data loaded successfully!{_RESET}")
        if self.debug:  
            print(f"Video start estimates: {self.video_start_estimates}")
        
        video = 0

        self.time_offsets = []
        for video_file in self.video_files:
            cap = cv2.VideoCapture(video_file)

            if not cap.isOpened():
                raise Exception(f"{_RED}Could not open video file: {video_file}{_RESET}")
    
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


            video_file_name = os.path.splitext(os.path.basename(video_file))[0]
            video_file_name = "." + video_file_name + ".csv"

            video_cache_csv_path = os.path.join(os.path.dirname(video_file), video_file_name).replace("\\", "/")
            if self.debug:
                print(f"Video file: {video_file}")
                print(f"Video cache CSV path: {video_cache_csv_path}")
            if os.path.exists(video_cache_csv_path):
                with open(video_cache_csv_path, 'r') as f:
                    reader = csv.reader(f)
                    april_tags = {}
                    for row in reader:
                        if len(row) < 3:
                            continue
                        if row[0] == '' or row[1] == '' or row[2] == '':
                            continue

                        april_tags[int(row[0])] = (float(row[1]), float(row[2]))
            else:
                april_tags = {}

            cap.set(cv2.CAP_PROP_POS_FRAMES, max(april_tags.keys(),default=-1) + 1)
            
            missing_frames = 0

            print(f"{_YELLOW}Processing video file: {video_file}{_RESET}")
            if self.debug:
                print(f"Total frames in video: {frame_count}")
                cv2.namedWindow('Drone Footage', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)

            while True:
                ret, frame = cap.read()
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                if frame_number >= frame_count:
                    break
                if not ret:
                    raise Exception(f"{_RED}Could not read frame from video file: {video_file}{_RESET}")   

                print(f"{_CYAN}Checking for AprilTags... {_BOLD}[{frame_number + 1}/{frame_count}]{_RESET}")

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                d = self.AprilTagDetector.detect(gray_frame)
                if d:
                    missing_frames = 0
                    if self.debug:
                        print(f"Detected AprilTag.")
                        print((d[0].center[0]-frame.shape[1] / 2) * -1,(d[0].center[1]-frame.shape[0] / 2))
                        print("ID", d[0].tag_id)

                    with open(video_cache_csv_path, "a") as f:
                        f.write(f"{frame_number},{(d[0].center[0]-frame.shape[1] / 2) * -1},{(d[0].center[1]-frame.shape[0] / 2)}\n")

                    april_tags[frame_number] = ((d[0].center[0]-frame.shape[1] / 2) * -1,(d[0].center[1]-frame.shape[0] / 2))
                else:
                    missing_frames += 1
                    if self.debug:
                        print(f"No AprilTags detected. {_BOLD}[{missing_frames}/{self.max_missing_frames}]{_RESET}")

                    with open(video_cache_csv_path, "a") as f:
                        f.write(f"{frame_number},,\n")

                    if missing_frames >= self.max_missing_frames: # Stops if two many frames are missing AprilTags in a row
                        print(f"{_RED}Too many missing frames, stopping detection.{_RESET}")
                        break
                
                if self.debug:
                    cv2.imshow('Drone Footage', frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print(f"{_YELLOW}Exiting video processing...{_RESET}")
                        break

                if frame_number >= frame_count - 1:
                    break

            cap.release()
            if self.debug:
                cv2.destroyWindow('Drone Footage')

            print(f"{_GREEN}AprilTag detection completed for video: {video_file}{_RESET}")

            aprilTags_t = np.array(list(april_tags.keys())) / self.fps


            # rotate the AprilTags coordinates based on the compass heading
            _c = math.cos(math.radians(self.compass_heading))
            _s = math.sin(math.radians(self.compass_heading))
            if april_tags:
                coords = np.array(list(april_tags.values()))
                __x = coords[:, 0]
                __y = coords[:, 1]
                aprilTags_x = __x * _c - __y * _s
                aprilTags_y = __x * _s + __y * _c

            if len(aprilTags_t) >= 2:
                self.aprilTags_x = CubicSpline(aprilTags_t, aprilTags_x)
                self.aprilTags_y = CubicSpline(aprilTags_t, aprilTags_y)
            else:
                raise Exception(f"{_RED}Not enough AprilTags detected in video: {video_file}{_RESET}")

            flight_csv_interpolated_t = np.arange(self.video_start_estimates[video], min(self.video_start_estimates[video] + aprilTags_t[-1] + 10, timestamps[-1]), self.timestep)
            video_interpolated_t = np.arange(0, aprilTags_t[-1], self.timestep)

            flight_positions = np.array([self.x(flight_csv_interpolated_t) - x[0], self.y(flight_csv_interpolated_t) - y[0]])  # subtracts the first value to make the starting home point (0,0)
            aprilTag_positions = np.array([self.aprilTags_x(video_interpolated_t), self.aprilTags_y(video_interpolated_t)])

            x_correlation = correlate(flight_positions[0], aprilTag_positions[0], mode='full')
            y_correlation = correlate(flight_positions[1], aprilTag_positions[1], mode='full')
            correlation = (x_correlation / np.max(x_correlation)) + (y_correlation / np.max(y_correlation))
            
            lags = correlation_lags(len(flight_positions[0]), len(aprilTag_positions[0]), mode='full') * self.timestep + self.video_start_estimates[video]
            self.time_offsets.append(lags[np.argmax(correlation)])
            print(f"{_GREEN}{self.time_offsets[video]} seconds for video {video_files[video]}{_RESET}")

            if self.debug:
                fig, axs = plt.subplots(2, 1, figsize=(10, 10))
                axs[0].plot(flight_csv_interpolated_t, flight_positions[0], label='Flight X', color='b')
                axs[0].plot([],[], label='Video X', color='r') # Placeholder for Video X
                axs[0].set_title('X Coordinates')
                axs[0].set_ylabel('X Coordinate (m)')
                axs[0].legend()

                ax2 = axs[0].twinx()
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('X Coordinate (px)')
                ax2.plot(video_interpolated_t + self.time_offsets[video], aprilTag_positions[0], label='Video X', color='r')

                align.yaxes(axs[0], 0, ax2, 0)


                axs[1].plot(flight_csv_interpolated_t, flight_positions[1], label='Flight Y', color='b')
                axs[1].plot([],[], label='Video Y', color='r')  # Placeholder for Video Y
                axs[1].set_title('Y Coordinates')
                axs[1].set_xlabel('Time (s)')
                axs[1].set_ylabel('Y Coordinate (m)')
                axs[1].legend()
                ax3 = axs[1].twinx()
                ax3.set_ylabel('Y Coordinate (px)')
                ax3.plot(video_interpolated_t + self.time_offsets[video], aprilTag_positions[1], label='Video Y', color='r')

                align.yaxes(axs[1], 0, ax3, 0)
                fig.tight_layout()
                fig.show()

                fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10))

                axs2[0].plot(lags, x_correlation / np.max(x_correlation), label='X Correlation', color='b')
                axs2[0].set_title('X Correlation')
                axs2[0].set_xlabel('Offset (s)')
                axs2[0].set_ylabel('Correlation Coefficient')
                axs2[0].legend()
                axs2[1].plot(lags, y_correlation / np.max(y_correlation), label='Y Correlation', color='b')
                axs2[1].set_title('Y Correlation')
                axs2[1].set_xlabel('Offset (s)')
                axs2[1].set_ylabel('Correlation Coefficient')
                axs2[1].legend()
                axs2[2].plot(lags, correlation, label='Correlation', color='b')
                axs2[2].set_title('Correlation')
                axs2[2].set_xlabel('Offset (s)')
                axs2[2].set_ylabel('Correlation Coefficient')
                axs2[2].axvline(self.time_offsets[video], color='r', linestyle='--', label='Time Offset')
                axs2[2].axvline(self.video_start_estimates[video], color='g', linestyle=':', label='Video Start Time Estimate')
                axs2[2].legend()
                fig2.tight_layout()
                fig2.show()

                plt.show()
            video += 1
    
    def get_pixel_on_video_from_position(self, video: int, timestamp, position: tuple[float, float]):
        T = np.array([])

    def get_position_from_video_timestamp(self, video: int, timestamp: float):
        return (float(self.x(timestamp + self.time_offsets[video])), float(self.y(timestamp + self.time_offsets[video])))
    
    def analyze(self, video: int):

        # watch every 10 frames
        
        cap = cv2.VideoCapture(self.video_files[video])
        if not cap.isOpened():
            raise Exception(f"{_RED}Could not open video file: {self.video_files[video]}{_RESET}")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        cv2.namedWindow('Drone Footage', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)  
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            timestamp = frame_number / self.fps
            cv2.circle(frame, self.get_pixel_on_video_from_position(video, timestamp, (628911.5959905937, 5516254.521853825)), 10, (0, 255, 0), -1)
            cv2.imshow('Drone Footage', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()