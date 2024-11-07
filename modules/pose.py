import math
import cv2
import numpy as np

from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.one_euro_filter import OneEuroFilter


class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    #added by Sita 07/11/2024
    #BODY_PARTS dictionary to map the keypoints to the corresponding index with a readability
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)

    #added by Sita 07/11/2024)
    
    def is_valid_point(self, point):
        #Check if the point is valid (not None and has valid coordinates).
        return point is not None and all(coord != -1 for coord in point)
    def calculate_angle(self, a, b, c):
        if a is None or b is None or c is None:
            return None
        
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
   
        a_length = np.linalg.norm(b-c)
        b_length = np.linalg.norm(a-c)
        c_length = np.linalg.norm(a-b)

        #use law of cosines to find angle C
        #cos(B) = (a^2 + c^2 - b^2) / 2ac
        angle = np.arccos((a_length**2 + c_length**2 - b_length**2) / (2 * a_length * c_length))
        return np.degrees(angle) #convert to degrees
                                     
    def draw_angles(self, img):
        # Get keypoints coordinates
        RShoulder = self.keypoints[self.BODY_PARTS["RShoulder"]]
        RElbow = self.keypoints[self.BODY_PARTS["RElbow"]]
        RWrist = self.keypoints[self.BODY_PARTS["RWrist"]]

        LShoulder = self.keypoints[self.BODY_PARTS["LShoulder"]]
        LElbow = self.keypoints[self.BODY_PARTS["LElbow"]]
        LWrist = self.keypoints[self.BODY_PARTS["LWrist"]]

        # Calculate angles
        r_arm_angle = self.calculate_angle(RShoulder, RElbow, RWrist)
        l_arm_angle = self.calculate_angle(LShoulder, LElbow, LWrist)

        # Right Arm
        if self.is_valid_point(RShoulder) and self.is_valid_point(RElbow) and self.is_valid_point(RWrist):
            cv2.line(img, tuple(RShoulder), tuple(RElbow), (0, 255, 0), 3)  
            cv2.line(img, tuple(RElbow), tuple(RWrist), (0, 255, 0), 3)  
            cv2.ellipse(img, tuple(RShoulder), (5, 5), 0, 0, 360, (0, 0, 255), -1)
            cv2.ellipse(img, tuple(RElbow), (5, 5), 0, 0, 360, (0, 0, 255), -1)
            cv2.ellipse(img, tuple(RWrist), (5, 5), 0, 0, 360, (0, 0, 255), -1)
            if r_arm_angle is not None:
                cv2.putText(img, f"{r_arm_angle:.2f}", tuple(RElbow), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Left Arm
        if self.is_valid_point(LShoulder) and self.is_valid_point(LElbow) and self.is_valid_point(LWrist):
            cv2.line(img, tuple(LShoulder), tuple(LElbow), (0, 255, 0), 3)
            cv2.line(img, tuple(LElbow), tuple(LWrist), (0, 255, 0), 3)  
            cv2.ellipse(img, tuple(LShoulder), (5, 5), 0, 0, 360, (0, 0, 255), -1)
            cv2.ellipse(img, tuple(LElbow), (5, 5), 0, 0, 360, (0, 0, 255), -1)
            cv2.ellipse(img, tuple(LWrist), (5, 5), 0, 0, 360, (0, 0, 255), -1)
            if l_arm_angle is not None:
                cv2.putText(img, f"{l_arm_angle:.2f}", tuple(LElbow), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)


