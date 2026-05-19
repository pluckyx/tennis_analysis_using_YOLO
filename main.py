from utils import read_video, save_video, measure_distance, draw_player_stats, convert_pixel_distance_to_meters, convert_meters_to_pixel_distance
from trackers import PlayerTracker, BallTracker
from court_lines import CourtLineDetector
from mini_court import MiniCourt
import cv2
import constants
import pandas as pd
from copy import deepcopy

def main():
    input_video_path = 'input_video/input_video.mp4'
    output_video_path = 'output_video/output_video.avi'
    
    video_frames = read_video(input_video_path)

    # Detect players & ball
    player_tracker = PlayerTracker('yolo26m.pt', pose_model_path='yolo26m-pose.pt')
    ball_tracker = BallTracker('models/best.pt')

    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect court keypoints (from first frame)
    court_model_path = 'models/coord.pt'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw mini court
    mini_court = MiniCourt(video_frames[0])

    # ball positions
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    #Convert positions to mini court
    # Filter players closest to court FIRST — before mini court coordinate conversion
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    #Convert positions to mini court (now only 2 players)
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)


    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,
        'player_1_last_hitting_hand': '-',

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
        'player_2_last_hitting_hand': '-',
    } ]

    # Map tracker IDs (e.g. 1, 3) → player numbers (1, 2) for stats keys
    # Get the two chosen player IDs from the first frame that has both players
    chosen_ids = []
    for frame_det in player_mini_court_detections:
        if len(frame_det) >= 2:
            chosen_ids = sorted(frame_det.keys())
            break
    if len(chosen_ids) < 2:
        chosen_ids = sorted(set(k for d in player_mini_court_detections for k in d.keys()))[:2]
    player_id_map = {chosen_ids[0]: 1, chosen_ids[1]: 2}

    for ball_shot_idx in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_idx]
        end_frame = ball_shot_frames[ball_shot_idx + 1]
        ball_shot_in_seconds = (end_frame - start_frame) / 24
        # ball_mini_court_detections is a list of dicts {1: (x,y)} — extract position with key 1
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_in_seconds * 3.6
        
        # Find which player hit the ball (closest to ball at start_frame)
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball_tracker_id = min(player_positions.keys(),
                               key=lambda pid: measure_distance(player_positions[pid],
                                                                ball_mini_court_detections[start_frame][1]))
        # Translate to player number 1 or 2
        player_shot_ball = player_id_map.get(player_shot_ball_tracker_id, 1)

        # The opponent is the other player number
        opponent_player_num = 2 if player_shot_ball == 1 else 1
        opponent_tracker_id = [tid for tid, num in player_id_map.items() if num == opponent_player_num]
        opponent_tracker_id = opponent_tracker_id[0] if opponent_tracker_id else None

        # Only compute opponent speed if they appear in both frames
        speed_of_opponent = 0
        if (opponent_tracker_id and
                opponent_tracker_id in player_mini_court_detections[start_frame] and
                opponent_tracker_id in player_mini_court_detections[end_frame]):
            distance_covered_by_opponent_pixels = measure_distance(
                player_mini_court_detections[start_frame][opponent_tracker_id],
                player_mini_court_detections[end_frame][opponent_tracker_id]
            )
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                distance_covered_by_opponent_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )
            speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_in_seconds * 3.6

        # --- Hitting hand detection using wrist keypoints ---
        # YOLO Pose keypoint indices: 9 = left wrist, 10 = right wrist
        hitting_hand = '-'
        shot_frame_data = player_detections[start_frame].get(player_shot_ball_tracker_id)
        if shot_frame_data and isinstance(shot_frame_data, dict):
            kps = shot_frame_data.get('keypoints')
            if kps is not None:
                # Get ball position in the original frame (use bounding box center as proxy)
                ball_box_at_shot = ball_detections[start_frame].get(1)
                if ball_box_at_shot:
                    ball_px, ball_py = (ball_box_at_shot[0]+ball_box_at_shot[2])/2, (ball_box_at_shot[1]+ball_box_at_shot[3])/2
                    left_wrist  = kps[9]   # (x, y)
                    right_wrist = kps[10]  # (x, y)
                    # Only use wrists that were actually detected (non-zero)
                    l_valid = left_wrist[0] > 0 and left_wrist[1] > 0
                    r_valid = right_wrist[0] > 0 and right_wrist[1] > 0
                    if l_valid and r_valid:
                        dist_left  = measure_distance(tuple(left_wrist),  (ball_px, ball_py))
                        dist_right = measure_distance(tuple(right_wrist), (ball_px, ball_py))
                        hitting_hand = 'Left' if dist_left < dist_right else 'Right'
                    elif l_valid:
                        hitting_hand = 'Left'
                    elif r_valid:
                        hitting_hand = 'Right'

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_hitting_hand'] = hitting_hand

        current_player_stats[f'player_{opponent_player_num}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_num}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']
    


    # Draw everything on video
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(output_video_frames, output_video_path)

if __name__ == '__main__':
    main()