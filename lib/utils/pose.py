import os
import cv2
import numpy as np
import pandas as pd
from ..config import *

import mediapipe as mp

def get_pose_estimation( video_path : str, with_info : bool = False) -> dict:
    """
    Uses MediaPipe detectors for pose and hand landmark detection and returns a dictionary containing the detected landmarks.
    Please visit:
        https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker
        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

    Parameters:
    -----------
        video_path : str
            path string for .mp4 file
        with_info: bool
            : logs `missing_left_hand` if MediaPipe could not detect left hand
            : logs `missing_right_hand` if MediaPipe could not detect right hand
            : logs `missing_pose` if MediaPipe could not detect pose
            : logs `total_number_of_frames` in given video
            : logs `configuration` parameters set for MediaPipe detectors

    Returns:
    --------
        landmarks : dict
            x,y,z coordinates of the detected landmarks as dictionary
            with the following keys `pose`, `right`, `left`;
            in total length of 75 (33 pose + 21 left hand + 21 right hand).
        info : list
            if `with_info` set True, returns a list with the following structure
            `video_name`, `info`, `value`
    """

    assert type(video_path) is str, f"{video_path} type must be 'str' ."
    assert video_path.endswith(".mp4"), f"{video_path} must end with '.mp4' ."
    assert os.path.exists(video_path), f"{video_path} does not exist !"

    video = cv2.VideoCapture(video_path)

    landmarks = {
        'pose': {
            'NOSE': [ ],
            'LEFT_EYE_INNER': [ ],
            'LEFT_EYE': [ ],
            'LEFT_EYE_OUTER': [ ],
            'RIGHT_EYE_INNER': [ ],
            'RIGHT_EYE': [ ],
            'RIGHT_EYE_OUTER': [ ],
            'LEFT_EAR': [ ],
            'RIGHT_EAR': [ ],
            'MOUTH_LEFT': [ ],
            'MOUTH_RIGHT': [ ],
            'LEFT_SHOULDER': [ ],
            'RIGHT_SHOULDER': [ ],
            'LEFT_ELBOW': [ ],
            'RIGHT_ELBOW': [ ],
            'LEFT_WRIST': [ ],
            'RIGHT_WRIST': [ ],
            'LEFT_PINKY': [ ],
            'RIGHT_PINKY': [ ],
            'LEFT_INDEX': [ ],
            'RIGHT_INDEX': [ ],
            'LEFT_THUMB': [ ],
            'RIGHT_THUMB': [ ],
            'LEFT_HIP': [ ],
            'RIGHT_HIP': [ ],
            'LEFT_KNEE': [ ],
            'RIGHT_KNEE': [ ],
            'LEFT_ANKLE': [ ],
            'RIGHT_ANKLE': [ ],
            'LEFT_HEEL': [ ],
            'RIGHT_HEEL': [ ],
            'LEFT_FOOT_INDEX': [ ],
            'RIGHT_FOOT_INDEX': [ ],
        },
        'right': {
            'WRIST': [ ],
            'THUMB_CMC': [ ],
            'THUMB_MCP': [ ],
            'THUMB_IP': [ ],
            'THUMB_TIP': [ ],
            'INDEX_FINGER_MCP': [ ],
            'INDEX_FINGER_PIP': [ ],
            'INDEX_FINGER_DIP': [ ],
            'INDEX_FINGER_TIP': [ ],
            'MIDDLE_FINGER_MCP': [ ],
            'MIDDLE_FINGER_PIP': [ ],
            'MIDDLE_FINGER_DIP': [ ],
            'MIDDLE_FINGER_TIP': [ ],
            'RING_FINGER_MCP': [ ],
            'RING_FINGER_PIP': [ ],
            'RING_FINGER_DIP': [ ],
            'RING_FINGER_TIP': [ ],
            'PINKY_MCP': [ ],
            'PINKY_PIP': [ ],
            'PINKY_DIP': [ ],
            'PINKY_TIP': [ ]
        },
        'left': {
            'WRIST': [ ],
            'THUMB_CMC': [ ],
            'THUMB_MCP': [ ],
            'THUMB_IP': [ ],
            'THUMB_TIP': [ ],
            'INDEX_FINGER_MCP': [ ],
            'INDEX_FINGER_PIP': [ ],
            'INDEX_FINGER_DIP': [ ],
            'INDEX_FINGER_TIP': [ ],
            'MIDDLE_FINGER_MCP': [ ],
            'MIDDLE_FINGER_PIP': [ ],
            'MIDDLE_FINGER_DIP': [ ],
            'MIDDLE_FINGER_TIP': [ ],
            'RING_FINGER_MCP': [ ],
            'RING_FINGER_PIP': [ ],
            'RING_FINGER_DIP': [ ],
            'RING_FINGER_TIP': [ ],
            'PINKY_MCP': [ ],
            'PINKY_PIP': [ ],
            'PINKY_DIP': [ ],
            'PINKY_TIP': [ ]
        }
    }

    # The MediaPipe Holistic Landmarker task lets you combine components of the 
    # pose, face, and hand landmarkers to create a complete landmarker for the human body.
    # Please visit:
    #   https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
    with mp.solutions.holistic.Holistic(
            static_image_mode=GLOBAL_CONFIG.MEDIAPIPE_STATIC_IMAGE_MODE,
            model_complexity=GLOBAL_CONFIG.MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks=GLOBAL_CONFIG.MEDIAPIPE_SMOOTH_LANDMARKS,
            enable_segmentation=GLOBAL_CONFIG.MEDIAPIPE_ENABLE_SEGMENTATION,
            smooth_segmentation=GLOBAL_CONFIG.MEDIAPIPE_SMOOTH_SEGMENTATION,
            refine_face_landmarks=GLOBAL_CONFIG.MEDIAPIPE_REFINE_FACE_LANDMARKS,
            min_detection_confidence=GLOBAL_CONFIG.MEDIAPIPE_MIN_DETECTION_CONFIDENCE, 
            min_tracking_confidence=GLOBAL_CONFIG.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        ) as holistic:
        
        
        # NOTE: There can be undetectable landmarks, log them if `with_info` set to True
        frame_idx = 0
        info = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            frame_idx = frame_idx + 1

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False

            # Convert the BGR image to RGB before processing.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame)

            if results.pose_landmarks:
                # Append pose landmarks to the list
                for landmark_name, landmark in zip(list(landmarks['pose'].keys()), results.pose_landmarks.landmark):
                    landmarks['pose'][landmark_name].append(np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]))
            else:
                if with_info:
                    info.append([f"{video_path}", "missing_pose", f"{frame_idx}"])
                    
                for landmark_name in landmarks['pose'].keys():
                    landmarks['pose'][landmark_name].append(np.array([
                        None,
                        None,
                        None,
                        None,
                    ]))

            if results.right_hand_landmarks:
                # Append right hand landmarks to the list
                for landmark_name, landmark in zip(list(landmarks['right'].keys()), results.right_hand_landmarks.landmark):
                    landmarks['right'][landmark_name].append(np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]))
            else:
                if with_info:
                    info.append([f"{video_path}", "missing_right_hand", f"{frame_idx}"])

                for landmark_name in landmarks['right'].keys():
                    landmarks['right'][landmark_name].append(np.array([
                        None,
                        None,
                        None,
                        None,
                    ]))

            if results.left_hand_landmarks:
                # Append left hand landmarks to the list
                for landmark_name, landmark in zip(list(landmarks['left'].keys()), results.left_hand_landmarks.landmark):
                    landmarks['left'][landmark_name].append(np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]))
            else:
                if with_info:
                    info.append([f"{video_path}", "missing_left_hand", f"{frame_idx}"])

                for landmark_name in landmarks['left'].keys():
                    landmarks['left'][landmark_name].append(np.array([
                        None,
                        None,
                        None,
                    ]))

    video.release()

    if with_info:
        info.append([f"{video_path}", "total_number_of_frames", f"{frame_idx}"])
        info.append([f"{video_path}", "configuration", f"{GLOBAL_CONFIG}"])
        return landmarks, info
    else:
        return landmarks, None
    

def get_pose_array(SAMPLE_POSE):
    """Converts the pose data into a numpy array
    """

    POSE_RAW = pd.DataFrame(SAMPLE_POSE['pose'])
    RIGHT_HAND_RAW = pd.DataFrame(SAMPLE_POSE['right'])
    LEFT_HAND_RAW = pd.DataFrame(SAMPLE_POSE['left'])

    POSE_DF = {}

    for col in POSE_RAW.columns:
        POSE_DF[ 'POSE_' + col + '_X'] = POSE_RAW[col].apply(lambda x: x[0])
        POSE_DF[ 'POSE_' + col + '_Y'] = POSE_RAW[col].apply(lambda x: x[1])
        POSE_DF[ 'POSE_' + col + '_Z'] = POSE_RAW[col].apply(lambda x: x[2])
        # POSE_DF[col + '_viz'] = POSE_RAW[col].apply(lambda x: x[3])

    for col in RIGHT_HAND_RAW.columns:
        POSE_DF[ 'RIGHT_' + col + '_X' ] = RIGHT_HAND_RAW[col].apply(lambda x: x[0])
        POSE_DF[ 'RIGHT_' + col + '_Y' ] = RIGHT_HAND_RAW[col].apply(lambda x: x[1])
        POSE_DF[ 'RIGHT_' + col + '_Z' ] = RIGHT_HAND_RAW[col].apply(lambda x: x[2])
        # POSE_DF['RIGHT_' + col + '_viz'] = RIGHT_HAND_RAW[col].apply(lambda x: x[3])

    for col in LEFT_HAND_RAW.columns:
        POSE_DF[ 'LEFT_' + col + '_X' ] = LEFT_HAND_RAW[col].apply(lambda x: x[0])
        POSE_DF[ 'LEFT_' + col + '_Y' ] = LEFT_HAND_RAW[col].apply(lambda x: x[1])
        POSE_DF[ 'LEFT_' + col + '_Z' ] = LEFT_HAND_RAW[col].apply(lambda x: x[2])
        # POSE_DF['LEFT_' + col + '_viz'] = LEFT_HAND_RAW[col].apply(lambda x: x[3])

    POSE_DF = pd.DataFrame(POSE_DF)

    return POSE_DF


def get_matrices(POSE_DF):
    """Converts the pose data into a numpy array of distance matrices
    """
    x_cols = [col for col in POSE_DF.columns if col.endswith('_X')]
    y_cols = [col for col in POSE_DF.columns if col.endswith('_Y')]
    z_cols = [col for col in POSE_DF.columns if col.endswith('_Z')]

    frames = []
    for i in range(1, POSE_DF.shape[0]):
        x_row = POSE_DF[x_cols].iloc[i].to_numpy()
        y_row = POSE_DF[y_cols].iloc[i].to_numpy()
        z_row = POSE_DF[z_cols].iloc[i].to_numpy()

        def get_difference_matrix(row):
            m, n = np.meshgrid(row, row)
            out = m-n
            return out

        x_diff = get_difference_matrix(x_row)
        y_diff = get_difference_matrix(y_row)
        z_diff = get_difference_matrix(z_row)

        frame = np.stack([x_diff, y_diff, z_diff], axis=2)
        frames.append(frame)

    frames = np.stack(frames, axis=0)
    return frames