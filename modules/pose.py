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
    
    '''
    The sigmas variable in the Pose class is an array that defines the standard deviations for Gaussian distributions used to model the uncertainty of the keypoints' positions. Each value corresponds to a specific body part's keypoint, indicating how much variation or noise can be expected around the detected position of that keypoint.
    '''
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]
    
    # Customer colors BGR
    kpts_colors = [255, 0, 0]
    

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
            # print("kpt_id:",kpt_id, keypoints[kpt_id])
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        # print("found_keypoints: ", bbox)
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
                cv2.circle(img, (int(x_a), int(y_a)), 6, Pose.kpts_colors, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 6, Pose.kpts_colors, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 5)

    def draw_skeleton(self, img, kpts_list = None):
        grey_value = 10  # Adjust this value to change the grey tone
        skeleton_color = (grey_value, grey_value, grey_value)  # BGR format for grey line color
        skeleton_color = tuple(list(skeleton_color[:3]) + [int(255 * 0.9)]) # Add transparency to the grey line color
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        
        if kpts_list is None:
            # video detection (draw the skeleton on the right side of the window)
            for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                global_kpt_a_id = self.keypoints[kpt_a_id, 0]
                if global_kpt_a_id != -1:
                    x_a, y_a = self.keypoints[kpt_a_id]
                    cv2.circle(img, (int(x_a), int(y_a)), 6, skeleton_color, -1)
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                global_kpt_b_id = self.keypoints[kpt_b_id, 0]
                if global_kpt_b_id != -1:
                    x_b, y_b = self.keypoints[kpt_b_id]
                    cv2.circle(img, (int(x_b), int(y_b)), 6, skeleton_color, -1)
                if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), skeleton_color, 50)
        else:
            # webcam real-time detection (draw the skeleton on the center of camera view)
            kpt_coords = {kpt['kpt_id']: tuple(kpt['coords']) for kpt in kpts_list}
            print("kpts_list: ", kpts_list)
            
            # scaling_factor = 1.35  # Adjust this value to increase or decrease the size

            # for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            #     kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            #     if kpt_a_id in kpt_coords:
            #         x_a, y_a = kpt_coords[kpt_a_id]
            #         x_a *= scaling_factor
            #         y_a *= scaling_factor
            #     kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            #     if kpt_b_id in kpt_coords:
            #         x_b, y_b = kpt_coords[kpt_b_id]
            #         x_b *= scaling_factor
            #         y_b *= scaling_factor
            #     if kpt_a_id in kpt_coords and kpt_b_id in kpt_coords:
            #         cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), skeleton_color, 80)
            
            # ======================================================================================================
            # Scaling and positioning the skeleton in the real-time webcam view
            scaling_factor = 1.35  # Adjust this value to increase or decrease the size
            scaled_keypoints = {}  # Dictionary to store scaled keypoints

            for kpt_id, (x, y) in kpt_coords.items():
                scaled_keypoints[kpt_id] = (x * scaling_factor, y * scaling_factor)

            # Calculate the centroid of the scaled body shape
            centroid_x = sum(x for x, y in scaled_keypoints.values()) / len(scaled_keypoints)
            centroid_y = sum(y for x, y in scaled_keypoints.values()) / len(scaled_keypoints)

            # Calculate the translation needed to move the centroid to the center of the OpenCV window
            window_center_x = img.shape[1] // 2
            window_center_y = img.shape[0] // 2
            window_bottom_y = img.shape[0] - 100 # Adjust the bottom vertical position (like a floor)
            translation_x = window_center_x - centroid_x
            translation_y = window_bottom_y - max(y for _, y in scaled_keypoints.values())

            # Update the coordinates based on the translation
            for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                x_a, y_a = scaled_keypoints.get(kpt_a_id, (0, 0))
                x_a += translation_x
                y_a += translation_y

                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                x_b, y_b = scaled_keypoints.get(kpt_b_id, (0, 0))
                x_b += translation_x
                y_b += translation_y

                if kpt_a_id in scaled_keypoints and kpt_b_id in scaled_keypoints:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), skeleton_color, 80)
                


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
