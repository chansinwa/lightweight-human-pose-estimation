import argparse

import cv2
import numpy as np
import torch
import os
import datetime
import shutil
import time
import psutil
import math
import threading

import threading
import pandas as pd
import json

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)  ## additional import for device configuration

skeleton_list = []
summary_report = {}
matching_kpts_report = []

## Load the JSON data from the file
with open('reports/tracking_frame_report_video-12.json', 'r') as file:
    skeleton_list = json.load(file)

def calculate_oks(pred_keypoints, gt_keypoints, area, keypoint_variance):
    """
    Calculate the Object Keypoint Similarity (OKS) between predicted and ground truth keypoints.

    :param pred_keypoints: Array of predicted keypoints (shape: K x 2)
    :param gt_keypoints: Array of ground truth keypoints (shape: K x 2)
    :param area: Area of the bounding box around the object
    :param keypoint_variance: Variance for each keypoint
    :return: OKS value
    """
    assert pred_keypoints.shape == gt_keypoints.shape
    K = pred_keypoints.shape[0]

    # Calculate distances for each keypoint
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)

    # Calculate OKS
    oks = np.sum(np.exp(-distances ** 2 / (2 * (keypoint_variance ** 2)))) / K
    return oks

# Function to calculate statistics of abs_distance
def calculate_distance_statistics(data):
    abs_distances = [item['abs_distance'] for sublist in data for item in sublist if item.get('abs_distance') is not None]
    max_distance = np.max(abs_distances)
    min_distance = np.min(abs_distances)
    mean_distance = np.mean(abs_distances)
    median_distance = np.median(abs_distances)
    std_distance = np.std(abs_distances)
    
    return max_distance, min_distance, mean_distance, median_distance, std_distance

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError("Image {} cannot be read".format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(
    net,
    img,
    net_input_height_size,
    stride,
    upsample_ratio,
    cpu,
    pad_value=(0, 0, 0),
    img_mean=np.array([128, 128, 128], np.float32),
    img_scale=np.float32(1 / 256),
):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(
        img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    # if not cpu:
    # tensor_img = tensor_img.cuda()
    tensor_img = tensor_img.to(device)  # Change here

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(
        heatmaps,
        (0, 0),
        fx=upsample_ratio,
        fy=upsample_ratio,
        interpolation=cv2.INTER_CUBIC,
    )

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(
        pafs,
        (0, 0),
        fx=upsample_ratio,
        fy=upsample_ratio,
        interpolation=cv2.INTER_CUBIC,
    )

    return heatmaps, pafs, scale, pad


# Function to write the info on the img, Tommy, 02-11-2024
def console_log(img, msg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    org = (10, 30)  # Coordinates for the top-left corner
    fontColor = (0, 255, 0)
    backgroundColor = (0, 0, 255)  # Red background color
    padding = 5
    lineType = 2  # Thickness of the line

    y = org[1]
    for key, value in msg.items():
        line = f"{key}: {value}"
        cv2.putText(img, line, (org[0], y), font, fontScale, fontColor, lineType)
        y += 20  # Adjust this value to control the spacing between lines

    return img



def run_demo(export_path, filename, net, image_provider, height_size, cpu, track, smooth, ref_kpts_list=None):      
    net = net.eval()
    # if not cpu:
    #     net = net.cuda()
    net = net.to(device)  # Change here

    fps_time = 0
    
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    
    tracking_frame_report = []
    keypoints_report = []
    
    frame_id = 0

    for img in image_provider:
        tracking_kpts_list = []
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(
            net, img, height_size, stride, upsample_ratio, cpu
        )

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]
            ) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    )
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    )
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            
        # print(current_poses)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        # Create a transparent image with the same dimensions as the original
        skeleton_img = np.zeros(
            (img.shape[0], img.shape[1], 4), dtype=np.uint8
        )  # Add an alpha channel
        skeleton_img[:, :, :3] = [255, 255, 255]  # Set RGB values to white color
        skeleton_img[:, :, 3] = 0  # Set alpha channel to 0 for full transparency
        
        # Combine the tracked image and the raw image on video tracking

        for pose in current_poses:
            cv2.rectangle(
                img,
                (pose.bbox[0], pose.bbox[1]),
                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
                (0, 255, 0),
            )
            if track:
                cv2.putText(
                    img,
                    "id: {}".format(pose.id),
                    (pose.bbox[0], pose.bbox[1] - 16),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                )

                for i, keypoint in enumerate(pose.keypoints):
                    kpt_id = i
                    kpt_name = Pose.kpt_names[i]
                    x, y = keypoint
                    
                    normalized_x = x / img.shape[1]
                    normalized_y = y / img.shape[0]
                    
                    tracking_kpts_list.append(
                        {
                            "frame_id": frame_id,
                            "kpt_id": kpt_id,
                            "kpt_name": kpt_name,
                            "img_size": img.shape[:2],
                            "coords": [x.tolist(), y.tolist()],
                            "normalized_coords": [normalized_x, normalized_y]
                        }
                    )  
                    
        # print("tracking_kpts_list: ", tracking_kpts_list)  
        keypoints_report.append(tracking_kpts_list)   
        
        for pose in current_poses:
            if ref_kpts_list is None:
                ### video detection
                pose.draw_skeleton(skeleton_img)
                pose.draw(img)
                pose.draw_angles(img)
            else:
                ### webcam real-time detection
                ## pass the 18 ref skeleton keypoints and webcam tracking keypoints of the current frame
                # print("ref_kpts_list:", [kpts['kpts'] for kpts in ref_kpts_list if kpts.get('frame_id') == frame_id], "\nwebcam_kpts_list:", tracking_kpts_list, "\n\n")
                result = pose.draw_skeleton(img, [kpts['kpts'] for kpts in ref_kpts_list if kpts.get('frame_id') == frame_id], [kpts for kpts in tracking_kpts_list if kpts.get('frame_id') == frame_id]) 
                
                if result is not None:
                    img_with_skeleton, combined_kpts = result
                else:
                    img_with_skeleton = img
                    combined_kpts = None
                
                if combined_kpts is not None and len(combined_kpts) > 0:
                    matching_kpts_report.append(combined_kpts[0])
                
                pose.draw(img_with_skeleton)
                pose.draw_angles(img_with_skeleton)
                pose.draw_colors_indicators(img_with_skeleton)
                
                img = img_with_skeleton
                
           
        # for pose in current_poses:
        #     # draw the checkpoints and lines on the img
        #     pose.draw(img)
        #     pose.draw_angles(img)
             
        ## Calculate the fps
        current_time = time.time()
        fps = round(1.0 / (current_time - fps_time), 2)
        
        ## Access the CPU usage
        current_cpu_load = psutil.cpu_percent()
                
        ## Write the info on the img, Tommy, 02-11-2024
        console_log(img, {"filename": filename, "frame_id": frame_id, "screen_size": img.shape[:2], "frame_time": current_time, "fps": fps, "cpu_load": current_cpu_load})
        
        ## Make the frame report
        tracking_frame_report.append({
            "filename": filename,
            "frame_id": frame_id,
            "frame_time": current_time,
            "fps": fps,
            "cpu_load": current_cpu_load,
            "kpts": tracking_kpts_list
        })
        
        image_name = "frame_" + str(frame_id) + ".jpg"
        
        ## Show the tracked image and skeleton image side by side
        if ref_kpts_list is None: 
            ## video detection
            img_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            combined_img = np.hstack((img_with_alpha, skeleton_img))
            cv2.imshow("Original and Skeleton", combined_img)
            
            ## Save the images
            cv2.imwrite(export_path + image_name, img)
            cv2.imwrite(export_path + "skt_" + image_name, skeleton_img)
        else: 
            ### webcam real-time detection
            # img_with_skeleton = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            cv2.imshow("Realtime webcam", img)
            ## Save the images
            cv2.imwrite(export_path + image_name, img)
        
        
                
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return keypoints_report, tracking_frame_report, matching_kpts_report
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

        frame_id += 1
        fps_time = time.time()
    
    
    return keypoints_report, tracking_frame_report, matching_kpts_report
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance."""
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoint_iter_370000.pth",
        help="path to the checkpoint",
    )
    parser.add_argument(
        "--height-size", type=int, default=256, help="network input layer height size"
    )
    parser.add_argument(
        "--video", type=str, default="", help="path to video file or camera id"
    )
    parser.add_argument(
        "--images", nargs="+", default="", help="path to input image(s)"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="run network inference on cpu"
    )
    parser.add_argument("--track", type=int, default=1, help="track pose id in video")
    parser.add_argument("--smooth", type=int, default=1, help="smooth pose keypoints")
    args = parser.parse_args()

    if args.video == "" and args.images == "":
        raise ValueError("Either --video or --image has to be provided")

    net = PoseEstimationWithMobileNet()
    # checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)  # change here
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != "":
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    ## create the export folder
    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    try:
        filename = os.path.basename(frame_provider.file_name)
    except:
        filename = "webcam"
    export_path = f"detection/exports/{filename}_{current_datetime}/"

    if not os.path.exists(f"{export_path}"):
        os.makedirs(f"{export_path}")
    else:
        shutil.rmtree(f"{export_path}/")
        os.makedirs(f"{export_path}/")

    print("imported file:", filename)
    

    start_time = time.time()
    print("Start processing...")
    result = None
    if args.video == '0' or args.video == '1' or args.video == '2':
        ### Real-time webcam detection
        result = run_demo(
            export_path,
            filename,
            net,
            frame_provider,
            args.height_size,
            args.cpu,
            args.track,
            args.smooth,
            skeleton_list
        )
    else:
        ### Video detection
        result = run_demo(
            export_path,
            filename,
            net,
            frame_provider,
            args.height_size,
            args.cpu,
            args.track,
            args.smooth
        )
        
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    ## Handle the reports
    print("result length:", len(result))
    keypoints_report, tracking_frame_report, matching_kpts_report = result
    
    avg_fps = sum([frame['fps'] for frame in tracking_frame_report]) / len(tracking_frame_report)
    avg_cpu_load = sum([frame['cpu_load'] for frame in tracking_frame_report]) / len(tracking_frame_report)
    
    if filename == 'webcam':
        max_distance, min_distance, mean_distance, median_distance, std_distance = calculate_distance_statistics(matching_kpts_report)
    
    summary_report = {
        "datetime": current_datetime,
        "filename": filename,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "avg_cpu_load": avg_cpu_load,
        "max_distance": max_distance if filename == 'webcam' else None,
        "min_distance": min_distance if filename == 'webcam' else None,
        "mean_distance": mean_distance if filename == 'webcam' else None,
        "median_distance": median_distance if filename == 'webcam' else None,
        "std_distance": std_distance if filename == 'webcam' else None
    }
    
    ## export JSON files
    # with open(f"{export_path}kpts_report.json", "w") as file:
    #     json.dump(keypoints_report, file, indent=4)
        
    with open(f"{export_path}tracking_frame_report.json", "w") as file:
        json.dump(tracking_frame_report, file, indent=4)
    
    with open(f"{export_path}summary_report.json", "w") as file:
        json.dump(summary_report, file, indent=4)
    
    with open(f"{export_path}matching_kpts_report.json", "w") as file:
        json.dump(matching_kpts_report, file, indent=4)
        
    # combined_frame_report = combine_frame_report(frame_report, webcam_frame_report)
    # with open(f"{export_path}combined_frame_report.json", 'w') as file:
    #     json.dump(combined_frame_report, file, indent=4)

    # Add this line to prevent the OpenCV windows from closing automatically
    cv2.waitKey(0)
