from ultralytics import YOLO 
import cv2
import pickle
import numpy as np
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path, pose_model_path='yolo26m-pose.pt'):
        self.model = YOLO(model_path)           # detection model
        self.pose_model = YOLO(pose_model_path)  # pose model (for cropped players)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, data in player_dict.items():
            bbox = data["bbox"] if isinstance(data, dict) else data
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks (or fewer if not enough detections)
        chosen_players = [d[0] for d in distances[:2]]
        return chosen_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    # COCO skeleton connections for drawing limbs
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # head
        (5, 6),                                   # shoulders
        (5, 7), (7, 9), (6, 8), (8, 10),         # arms
        (5, 11), (6, 12),                         # torso
        (11, 12),                                  # hips
        (11, 13), (13, 15), (12, 14), (14, 16)    # legs
    ]

    def detect_frame(self, frame):
        """
        Two-stage detection pipeline:
        
        1. Why we do this:
           - The YOLO Pose model struggles to detect small/distant people (like the far tennis player) 
             because predicting 17 keypoints + a bounding box at a distance is too complex.
           - The regular YOLO detection model (no pose) is much better at finding small people.
        
        2. How it works:
           - STAGE 1 (Detection): We use the regular YOLO model at a high resolution (1920) 
             to reliably find all people in the frame (including the far player).
           - STAGE 2 (Pose): For each detected person, we crop their exact bounding box from the frame. 
             Since the player now fills this cropped image, the Pose model can easily detect their 
             joints, even if they were far away originally. Finally, we map these joint coordinates 
             back to the original full-frame coordinates.
        """
        # Stage 1: Detect with regular model
        # 'results' contains bounding boxes, classes, and confidence scores for the whole frame.
        results = self.model.track(frame, persist=True, conf=0.15, imgsz=1920)[0]
        # 'id_name_dict' maps class IDs to their string names (e.g., 0 -> "person")
        id_name_dict = results.names

        player_dict = {}
        for i, box in enumerate(results.boxes):
            if box.id is None:
                continue
            # 'track_id' is the unique ID given to this person by the YOLO tracker across frames
            track_id = int(box.id.tolist()[0])
            # 'result' is the bounding box coordinates [x1, y1, x2, y2]
            result = box.xyxy.tolist()[0]
            # 'object_cls_id' is the integer class ID
            object_cls_id = box.cls.tolist()[0]
            # 'object_cls_name' is the string name (we only want "person")
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                # We save the bounding box and initialize 'keypoints' as None
                player_dict[track_id] = {"bbox": result, "keypoints": None}

        # Stage 2: Run pose model on each cropped player
        for track_id, data in player_dict.items():
            # Extract the raw bounding box coordinates of the detected person
            x1, y1, x2, y2 = [int(v) for v in data["bbox"]]
            
            # 'h' (height) and 'w' (width) of the original full video frame
            h, w = frame.shape[:2]
            
            # 'pad' adds 20 pixels around the person. If the crop is too tight, the pose model might miss limbs.
            pad = 20
            
            # Calculate the new padded crop coordinates (cx1, cy1, cx2, cy2).
            # max(0, ...) and min(w/h, ...) ensure we don't accidentally try to crop outside the boundaries of the image.
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            
            # 'crop' is the small image cutout containing ONLY the player (and the padding)
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            # Run the pose model on ONLY the 'crop' image.
            pose_results = self.pose_model.predict(crop, conf=0.3, verbose=False)[0]
            
            if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                # 'kps' is an array of shape (17, 2). It holds the 17 [x, y] coordinates for the body joints.
                # IMPORTANT: These [x, y] coordinates are relative to the TOP-LEFT corner of the 'crop' image, 
                # NOT the full video frame!
                kps = pose_results.keypoints[0].xy.cpu().numpy()[0]  # shape (17, 2)
                
                # Map keypoints back to original frame coordinates.
                # Since the crop starts at (cx1, cy1) in the main image, we simply add cx1 to all X coordinates
                # and add cy1 to all Y coordinates of the joints to place them correctly on the full video frame.
                kps[:, 0] += cx1
                kps[:, 1] += cy1
                
                # Store the corrected keypoints back into our dictionary
                data["keypoints"] = kps

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for player_num, (track_id, data) in enumerate(player_dict.items(), start=1):
                # Handle both old format (just bbox list) and new format (dict with bbox + keypoints)
                if isinstance(data, dict):
                    bbox = data["bbox"]
                    kps = data.get("keypoints")
                else:
                    bbox = data
                    kps = None

                x1, y1, x2, y2 = bbox
                # Always label as Player 1 / Player 2 regardless of the internal tracker ID
                cv2.putText(frame, f"Player {player_num}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Draw pose skeleton
                if kps is not None:
                    # Draw keypoint dots
                    for j, (px, py) in enumerate(kps):
                        if px > 0 and py > 0:
                            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

                    # Draw skeleton lines
                    for (a, b) in self.SKELETON:
                        if kps[a][0] > 0 and kps[a][1] > 0 and kps[b][0] > 0 and kps[b][1] > 0:
                            cv2.line(frame, (int(kps[a][0]), int(kps[a][1])),
                                     (int(kps[b][0]), int(kps[b][1])), (0, 255, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames