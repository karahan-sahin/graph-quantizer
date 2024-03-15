import cv2
import mediapipe as mp
import numpy as np

def estimate_pose_and_save(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.holistic

    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize pose estimation
    pose_data = []
    left_hand_data = []
    right_hand_data = []
    with mp_pose.Holistic(min_detection_confidence=0.01, min_tracking_confidence=0.01) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose estimation
            results = pose.process(image)

            # Save pose data into a list
            if results.pose_landmarks is not None:
                pose_data.append([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark])
                left_hand_data.append([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.left_hand_landmarks.landmark])
            else:
                pose_data.append([])

            # Display the image
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Convert the list to a numpy array and save it
    pose_data = np.array(pose_data)
    np.save('pose_data.npy', pose_data)

# Call the function with the path to your video file
if __name__ == '__main__':
    estimate_pose_and_save('/path/to/your/video/file')