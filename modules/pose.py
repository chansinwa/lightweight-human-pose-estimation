import cv2
import numpy as np
import math

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
    kpts_colors = [0, 225, 255]
    line_color = [255, 255, 0]
    skeleton_overlay_color = [10, 10, 10]
    skeleton_line_color = [255, 0, 255]
    
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

        ## Draw the tracking lines
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 6, Pose.kpts_colors, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            # if global_kpt_b_id != -1:
            #     x_b, y_b = self.keypoints[kpt_b_id]
            #     cv2.circle(img, (int(x_b), int(y_b)), 6, Pose.kpts_colors, -1)
                
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.line_color, 5)
        
        ## Draw the keypoints
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
            
    
    def draw_skeleton(self, img, ref_kpts_list = None, webcam_kpts_list=None):       
        # grey_value = 10  # Adjust this value to change the grey tone
        # skeleton_color = (grey_value, grey_value, grey_value, int(255 * 0.9))  # BGRA format with alpha for transparency

        assert self.keypoints.shape == (Pose.num_kpts, 2)
        
        if ref_kpts_list is None:
            ### video detection (draw the skeleton on the right side of the window)
            for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                global_kpt_a_id = self.keypoints[kpt_a_id, 0]
                if global_kpt_a_id != -1:
                    x_a, y_a = self.keypoints[kpt_a_id]
                    cv2.circle(img, (int(x_a), int(y_a)), 6, Pose.skeleton_overlay_color, -1)
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                global_kpt_b_id = self.keypoints[kpt_b_id, 0]
                if global_kpt_b_id != -1:
                    x_b, y_b = self.keypoints[kpt_b_id]
                    cv2.circle(img, (int(x_b), int(y_b)), 6, Pose.skeleton_overlay_color, -1)
                if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.skeleton_overlay_color, 50)
            
        else:
            ### webcam real-time detection (draw the skeleton on the center of camera view)
            ## Extract the 18 normalized coordinates from the imported list and create this object: {0: (normalized_x, normalized_y), ...}
            if len(ref_kpts_list) > 0:
                scaled_kpt = {}
                circle_coordinates = []
                circle_colors = []
                
                for kpt in ref_kpts_list[0]:
                    if 'normalized_coords' in kpt and kpt['coords'] != [-1, -1]:
                        kpt_id = kpt['kpt_id']
                        x, y = kpt['normalized_coords']
                        ## fit the normalized coordinates to real-time size (1920x1080)
                        x *= img.shape[1]
                        y *= img.shape[0]
                        scaled_kpt[kpt_id] = [x, y]
                        kpt['ref_scaled_coords'] = scaled_kpt[kpt_id]
                        
                        # Change the key name from 'coords' to 'ref_coords'
                        kpt['ref_coords'] = kpt.pop('coords')
                        kpt['ref_normalized_coords'] = kpt.pop('normalized_coords')
                        
                        
                        ## Calculate the abs distance between the keypoints
                        abs_distance = 0
                        if webcam_kpts_list is not None:
                            webcam_kpt_corrds = webcam_kpts_list[kpt_id]['coords']
                            abs_distance = self.calculate_distance(kpt['ref_scaled_coords'], webcam_kpt_corrds)
                            # print("fram_id", kpt['frame_id'], "kpt_id: ", kpt_id, "ref_scaled_coords: ", kpt['ref_scaled_coords'], "webcam_kpt_corrds: ", webcam_kpt_corrds, "abs_distance: ", abs_distance)
                        
                            kpt['webcam_coords'] = webcam_kpts_list[kpt_id]['coords']
                            kpt['abs_distance'] = abs_distance
                            
                            if abs_distance <= 80:
                                circle_coordinates.append((int(x), int(y)))
                                circle_colors.append((0, 255, 0))  # Green color for correct posture
                            else:
                                circle_coordinates.append((int(x), int(y)))
                                circle_colors.append((0, 0, 255))  # Red color for incorrect posture
                        
                        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
                            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                            x_a, y_a = scaled_kpt.get(kpt_a_id, (0, 0))

                            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                            x_b, y_b = scaled_kpt.get(kpt_b_id, (0, 0))

                            if kpt_a_id in scaled_kpt and kpt_b_id in scaled_kpt:
                                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.skeleton_overlay_color, 80) 
                        
                        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
                            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                            x_a, y_a = scaled_kpt.get(kpt_a_id, (0, 0))

                            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                            x_b, y_b = scaled_kpt.get(kpt_b_id, (0, 0))

                            if kpt_a_id in scaled_kpt and kpt_b_id in scaled_kpt:
                                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.skeleton_line_color, 5)
                                 
                        
                for coord, color in zip(circle_coordinates, circle_colors):
                    cv2.circle(img, coord, 15, color, -1)
                
                self.draw_colors_indicators(img)
            else:
                return
        # print("updated kpts_list: ", kpts_list) 
        
        return ref_kpts_list

    def draw_colors_indicators(self, img):
        color_mapping = {
            (255, 0, 255): "Reference motion",
            (255, 255, 0): "Real-time motion",
            (0, 255, 0): "Correct joint posture",
            (0, 0, 255): "Incorrect joint posture"
        }

        indicator_size = 30  # Size of the color indicator square
        text_offset = 10  # Offset for the text

        ## Define the bottom left corner coordinates for drawing
        x_start = 20
        y_start = img.shape[0] - 180  # Adjust as needed

        for idx, (color, label) in enumerate(color_mapping.items()):
            ## Draw the color square
            cv2.rectangle(img, (x_start, y_start + idx * (indicator_size + text_offset)),
                          (x_start + indicator_size, y_start + indicator_size + idx * (indicator_size + text_offset)),
                          color, -1)

            ## Draw the color name
            cv2.putText(img, label, (x_start + indicator_size + 10, y_start + indicator_size + idx * (indicator_size + text_offset) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        return img
        
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
        ## use law of cosines to find angle C
        ## cos(B) = (a^2 + c^2 - b^2) / 2ac
        angle = np.arccos((a_length**2 + c_length**2 - b_length**2) / (2 * a_length * c_length))
        return np.degrees(angle) #convert to degrees

    def calculate_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def draw_text_with_outline(self, img, text, position, font_scale, thickness):
        ## Draw text with an outline effect (black outline (border)).
        cv2.putText(img, text, (position[0]-1, position[1]-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, text, (position[0]+1, position[1]-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, text, (position[0]-1, position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, text, (position[0]+1, position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)

        ## Draw the red text on top
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    
    ## Old function to draw the angle sector
    def draw_angle_sector(self, img, center, line1, line2, radius=50):
        # Draw a circular sector to represent the angle visually.
        # angle = self.calculate_angle(line1, center, line2)
        start_angle = np.degrees(np.arctan2(line1[1] - center[1], line1[0] - center[0]))
        end_angle = np.degrees(np.arctan2(line2[1] - center[1], line2[0] - center[0]))
        # end_angle = start_angle + angle
        
        ## Ensure the angles are in the correct order
        if end_angle < start_angle:
            end_angle += 360
        #     print("end_angle", end_angle, "start_angle", start_angle)
            
        #     if ((end_angle % 360) - (start_angle % 360))  > 180:
        #         start_angle, end_angle = end_angle, start_angle  # Swap the angles
        
        print("start_angle", start_angle, "end_angle", end_angle)
                
        # start_angle = start_angle % 360
        # end_angle = end_angle % 360
        # Draw the arc
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, (255, 0, 0), 2)

    def draw_angles(self, img):
        ## Get keypoints coordinates
        RShoulder = self.keypoints[self.BODY_PARTS["RShoulder"]]
        RElbow = self.keypoints[self.BODY_PARTS["RElbow"]]
        RWrist = self.keypoints[self.BODY_PARTS["RWrist"]]

        LShoulder = self.keypoints[self.BODY_PARTS["LShoulder"]]
        LElbow = self.keypoints[self.BODY_PARTS["LElbow"]]
        LWrist = self.keypoints[self.BODY_PARTS["LWrist"]]
        
        Neck = self.keypoints[self.BODY_PARTS["Neck"]]
        RShoulder = RShoulder
        RElbow = RElbow
        
        Neck = Neck
        LShoulder = LShoulder
        LElbow = LElbow
        
        Neck = Neck
        RHip = self.keypoints[self.BODY_PARTS["RHip"]]
        RAnkle = self.keypoints[self.BODY_PARTS["RAnkle"]]
        
        Neck = Neck
        LHip = self.keypoints[self.BODY_PARTS["LHip"]]
        LAnkle = self.keypoints[self.BODY_PARTS["LAnkle"]]

        ## Calculate angles
        r_arm_angle = self.calculate_angle(RShoulder, RElbow, RWrist)
        l_arm_angle = self.calculate_angle(LShoulder, LElbow, LWrist)
        r_shoulder_angle = self.calculate_angle(Neck, RShoulder, RElbow)
        l_shoulder_angle = self.calculate_angle(Neck, LShoulder, LElbow)
        r_hip_angle = self.calculate_angle(Neck, RHip, RAnkle)
        l_hip_angle = self.calculate_angle(Neck, LHip, LAnkle)
        

        ## Calculate distance for dynamic angle value text size
        distance_r_arm = self.calculate_distance(RElbow, RWrist)
        font_scale_r = min(max(distance_r_arm / 100, 0.5), 0.5)
        distance_l_arm = self.calculate_distance(LElbow, LWrist)
        font_scale_l = min(max(distance_l_arm / 100, 0.5), 0.5)
        distance_r_shoulder = self.calculate_distance(RShoulder, RElbow)
        font_scale_r_shoulder = min(max(distance_r_shoulder / 100, 0.5), 0.5)
        distance_l_shoulder = self.calculate_distance(LShoulder, LElbow)
        font_scale_l_shoulder = min(max(distance_l_shoulder / 100, 0.5), 0.5)
        distance_r_hip = self.calculate_distance(RHip, RAnkle)
        font_scale_r_hip = min(max(distance_r_hip / 100, 0.5), 0.5)
        distance_l_hip = self.calculate_distance(LHip, LAnkle)
        font_scale_l_hip = min(max(distance_l_hip / 100, 0.5), 0.5)

        ## Right Arm
        if self.is_valid_point(RShoulder) and self.is_valid_point(RElbow) and self.is_valid_point(RWrist):
            if r_arm_angle is not None:
                # self.draw_angle_sector(img, tuple(RElbow), RShoulder, RWrist)
                self.draw_interior_sector(img, tuple(RElbow), tuple(RShoulder), tuple(RWrist), "right") 
                # cv2.putText(img, f"{r_arm_angle:.2f}", tuple(RElbow), cv2.FONT_HERSHEY_SIMPLEX, font_scale_r, (0, 0, 0), 2)
                self.draw_text_with_outline(img, f"{r_arm_angle:.1f}", tuple(RElbow), font_scale_r, 1)
                
        ## Left Arm
        if self.is_valid_point(LShoulder) and self.is_valid_point(LElbow) and self.is_valid_point(LWrist):
            if l_arm_angle is not None:
                # self.draw_angle_sector(img, tuple(LElbow), LShoulder, LWrist) 
                self.draw_interior_sector(img, tuple(LElbow), tuple(LShoulder), tuple(LWrist), "left")
                # cv2.putText(img, f"{l_arm_angle:.2f}", tuple(LElbow), cv2.FONT_HERSHEY_SIMPLEX, font_scale_l, (0, 0, 0), 2)
                self.draw_text_with_outline(img, f"{l_arm_angle:.1f}", tuple(LElbow), font_scale_l, 1)
        
        ## Right Shoulder
        if self.is_valid_point(Neck) and self.is_valid_point(RShoulder) and self.is_valid_point(RElbow):
            if r_shoulder_angle is not None:
                # self.draw_angle_sector(img, tuple(RShoulder), Neck, RElbow)
                self.draw_interior_sector(img, tuple(RShoulder), Neck, RElbow, "right")
                # cv2.putText(img, f"{r_shoulder_angle:.2f}", tuple(RShoulder), cv2.FONT_HERSHEY_SIMPLEX, font_scale_r_shoulder, (0, 0, 0), 2)
                self.draw_text_with_outline(img, f"{r_shoulder_angle:.1f}", tuple(RShoulder), font_scale_r_shoulder, 1)
        
        ## Left Shoulder
        if self.is_valid_point(Neck) and self.is_valid_point(LShoulder) and self.is_valid_point(LElbow):
            if l_shoulder_angle is not None:
                # self.draw_angle_sector(img, tuple(LShoulder), Neck, LElbow)
                self.draw_interior_sector(img, tuple(LShoulder), Neck, LElbow, "left")
                # cv2.putText(img, f"{l_shoulder_angle:.2f}", tuple(LShoulder), cv2.FONT_HERSHEY_SIMPLEX, font_scale_l_shoulder, (0, 0, 0), 2)
                self.draw_text_with_outline(img, f"{l_shoulder_angle:.1f}", tuple(LShoulder), font_scale_l_shoulder, 1)
        
        ## Right Hip
        if self.is_valid_point(Neck) and self.is_valid_point(RHip) and self.is_valid_point(RAnkle):
            if r_hip_angle is not None:
                # self.draw_angle_sector(img, tuple(RHip), Neck, RAnkle)
                self.draw_interior_sector(img, tuple(RHip), Neck, RAnkle, "right")
                # cv2.putText(img, f"{r_hip_angle:.2f}", tuple(RHip), cv2.FONT_HERSHEY_SIMPLEX, font_scale_r_hip, (0, 0, 0), 2)
                self.draw_text_with_outline(img, f"{r_hip_angle:.1f}", tuple(RHip), font_scale_r_hip, 1)
        
        ## Left Hip
        if self.is_valid_point(Neck) and self.is_valid_point(LHip) and self.is_valid_point(LAnkle):
            if l_hip_angle is not None:
                # self.draw_angle_sector(img, tuple(LHip), Neck, LAnkle)
                self.draw_interior_sector(img, tuple(LHip), Neck, LAnkle, "left")
                # cv2.putText(img, f"{l_hip_angle:.2f}", tuple(LHip), cv2.FONT_HERSHEY_SIMPLEX, font_scale_l_hip, (0, 0, 0), 2)
                self.draw_text_with_outline(img, f"{l_hip_angle:.1f}", tuple(LHip), font_scale_l_hip, 1)
        
    
    ## New function to calculate the angle for drawing the sector on the interior angle side, Tommy, 05-01-2024            
    def calculate_clockwise_angle_from_x_axis(self, center, pt):
        angle_radian: float = 0
        angle_deg: float = 0
            
        if (pt[0] >= center[0] and pt[1] > center[1]): # Quadrant I
            angle_radian = math.atan((pt[1] - center[1]) / (pt[0] - center[0])) # arctan(y/x)
            angle_deg = angle_radian * 180 / math.pi # Convert to degrees
        elif (pt[0] < center[0] and pt[1] >= center[1]): # Quadrant II
            angle_radian = math.atan((center[0] - pt[0]) / (pt[1] - center[1])) # arctan(x/y)
            angle_deg = 90 + angle_radian * 180 / math.pi # Convert to degrees
        elif (pt[0] <= center[0] and pt[1] < center[1]): # Quadrant III
            angle_radian = math.atan((center[1] - pt[1]) / (center[0] - pt[0])) # arctan(y/x)
            angle_deg = 180 + angle_radian * 180 / math.pi # Convert to degrees
        elif (pt[0] > center[0] and pt[1] <= center[1]): # Quadrant IV
            angle_radian = math.atan((pt[0] - center[0]) / (center[1] - pt[1])) # arctan(x/y)
            angle_deg = 270 + angle_radian * 180 / math.pi # Convert to degrees

        return angle_deg
    
    ## New function to draw the sector on the interior angle side, Tommy, 05-01-2024  
    def draw_interior_sector(self, img, center, pt_a, pt_b, body_part = None):
        x_axis_angle: float = 0
        end_angle: float = 0
        radius = 30
        arc_color = (0, 127, 255)
        sector_color = (0, 127, 255)
        
        angle_a = self.calculate_clockwise_angle_from_x_axis(center, pt_a)
        angle_b = self.calculate_clockwise_angle_from_x_axis(center, pt_b)
        
        if angle_b < angle_a:
            temp = angle_a
            angle_a = angle_b
            angle_b = temp
            
        if angle_b - angle_a < 180:
            x_axis_angle = angle_a
            end_angle = angle_b - angle_a
        else:
            x_axis_angle = angle_b
            end_angle = 360 - (angle_b - angle_a)
        
        # print("Center: ", center, "Point A: ", pt_a, "Point B: ", pt_b ,"X-axis angle: ", x_axis_angle, "End angle: ", end_angle)
        
        """
        ## Determine the sector direction based on the body part
        if body_part == "left":
            if angle_a < angle_b:  # Ensure sector points towards the left
                x_axis_angle = angle_b
            else:
                x_axis_angle = angle_a
        elif body_part == "right":
            if angle_a < angle_b:  # Ensure sector points towards the right
                x_axis_angle = angle_a
            else:
                x_axis_angle = angle_b
        """
        
        ## Draw the arc border line (start andgle always 0)
        cv2.ellipse(img, center, (radius, radius), x_axis_angle, 0, end_angle, arc_color, 2)
        
        ## Draw the filled sector
        transparency = 0.4
        original_img = img.copy()
        cv2.ellipse(img, center, (radius, radius), x_axis_angle, 0, end_angle, sector_color, -1)
        cv2.addWeighted(img, transparency, original_img, 1 - transparency, 0, img)




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
