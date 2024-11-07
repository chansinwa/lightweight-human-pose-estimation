import argparse

import cv2
import numpy as np
import torch
import os
import datetime
import shutil
import time

import threading
import pandas as pd
import json

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

# change here
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)  # additional import for device configuration


# Load the JSON data from the file
skeleton_list = []
with open('kpts_list.json', 'r') as file:
    skeleton_list = json.load(file)


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

    lines = [
        "inputed file: " + str(msg["filename"]),
        "frame: " + str(msg["frame_num"]),
    ]

    y = org[1]
    for line in lines:
        # (text_width, text_height), _ = cv2.getTextSize(line, font, fontScale, lineType)
        # cv2.rectangle(img, (org[0], y - text_height + padding), (org[0] + text_width + 2 * padding, y + padding), backgroundColor, -1) # Draw text background
        cv2.putText(img, line, (org[0], y), font, fontScale, fontColor, lineType)
        y += 20  # Adjust this value to control the spacing between lines

    return img


def run_demo(net, image_provider, height_size, cpu, track, smooth, ref_ckpt_list=None):      
    net = net.eval()
    # if not cpu:
    #     net = net.cuda()
    net = net.to(device)  # Change here

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    # code for saving files, Tommy, 02-11-2024
    frame_num = 0
    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    try:
        filename = os.path.basename(image_provider.file_name)
    except:
        filename = "webcam"
    export_path = f"detection/exports/{filename}_{current_datetime}/"

    if not os.path.exists(f"{export_path}"):
        os.makedirs(f"{export_path}")
    else:
        shutil.rmtree(f"{export_path}/")
        os.makedirs(f"{export_path}/")

    print("imported file:", filename)
    # code for saving files

    # Initialize FPS calculation variables
    start_time = time.time()
    frame_count = 0
    fps = 0

    keypoints_info = []

    for img in image_provider:
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
            # list of keypoints data
            # print("pose:", pose_keypoints)

        # print(current_poses)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        for pose in current_poses:
            # draw the checkpoints and lines on the img
            pose.draw(img)

        # Create a transparent image with the same dimensions as the original
        skeleton_img = np.zeros(
            (img.shape[0], img.shape[1], 4), dtype=np.uint8
        )  # Add an alpha channel
        skeleton_img[:, :, 3] = 0  # Set alpha channel to 0 for full transparency

        for pose in current_poses:
            if ref_ckpt_list is None:
                # video detection
                pose.draw_skeleton(skeleton_img)
            else:
                # webcam real-time detection
                pose.draw_skeleton(img, [ckpt for ckpt in ref_ckpt_list if ckpt.get('frame_id') == frame_num]) # pass the 18 skeleton keypoints of the current frame

        # Replace the white background with transparency
        skeleton_img[skeleton_img[:, :, 3] == 255] = [0, 0, 0, 0]

        # Combine the tracked image and the raw image on video tracking
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

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

                # print("pose.keypoints:", pose.keypoints)
                # Create a new array of objects with the desired format, for printing the keypoints data
                # making the keypoint_list
                for i, keypoint in enumerate(pose.keypoints):
                    kpt_id = i
                    kpt_name = Pose.kpt_names[i]
                    x, y = keypoint
                    keypoints_info.append(
                        {
                            "frame_id": frame_num,
                            "kpt_id": kpt_id,
                            "kpt_name": kpt_name,
                            "coords": [x, y],
                        }
                    )

                # Print the new array of objects in the desired format
                # for info in keypoints_info:
                #     print(info)

                # # Create a DataFrame from the keypoints_info list
                # df = pd.DataFrame(keypoints_info)
                # # Export the DataFrame to an Excel file
                # df.to_excel(export_path + "keypoints_info.xlsx", index=False)

                # fps_time = time.time()
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:  # Calculate and display FPS every 10 frames
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time

                # Write the info on the img, Tommy, 02-11-2024
                console_log(img, {"filename": filename, "frame_num": frame_num})
                # Save the image, Tommy, 02-11-2024
                image_name = "frame_" + str(frame_num) + ".jpg"
                
                # this two lines are for saving images
                cv2.imwrite(export_path + image_name, img)
                cv2.imwrite(export_path + "skt_" + image_name, skeleton_img)

        # cv2.imshow("Lightweight Human Pose Estimation Python Demo", img)
        # cv2.imshow("Skeleton", skeleton_img)

        # Show the tracked image and skeleton image side by side
        if ref_ckpt_list is None: 
            # video detection
            img_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            combined_img = np.hstack((img_with_alpha, skeleton_img))
            cv2.imshow("Original and Skeleton", combined_img)
        else: 
            # webcam real-time detection
            cv2.imshow("Realtime webcam", img)
        
        # Resize the skeleton image to match the dimensions of the original image
        # skeleton_img_resized = cv2.resize(skeleton_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

        # Blend the original image and the skeleton image
        # final_img = cv2.addWeighted(img, 0.8, skeleton_img_resized, 0.2, 0)
        # cv2.imshow(final_img)

        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

        frame_num += 1

    # print the whole keypoints_info list
    # print("kpts_list:", keypoints_info)
    
    # this is for saving the text file of the checkpoint list
    with open(f"{export_path}_kpts.txt", "w") as file:
        file.write(str(keypoints_info))


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

        # run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth) # video
        # Define the functions to run the demos

    if args.video == '0' or args.video == '1' or args.video == '2':
        run_demo(
            net,
            frame_provider,
            args.height_size,
            args.cpu,
            args.track,
            args.smooth,
            skeleton_list
        )
    else:
        run_demo(
            net,
            frame_provider,
            args.height_size,
            args.cpu,
            args.track,
            args.smooth
        )
    
    # Add this line to prevent the OpenCV windows from closing automatically
    cv2.waitKey(0)
