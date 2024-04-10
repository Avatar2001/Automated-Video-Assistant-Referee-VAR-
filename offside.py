import argparse
import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np
import os

from sklearn.cluster import KMeans
from skimage.color import rgb2lab

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
with open('hyp.scratch.mask.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)

def load_model():
    model = torch.load('yolov7-mask.pt', map_location=device)['model']
    # Convert model weights to float32
    model.float()
    # Put in inference mode
    model.eval()

    if torch.cuda.is_available():
        model.to(device)
    else:
        model.to(torch.device('cpu'))

    return model

model = load_model()


def remove_background(image_path):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV, assuming the field is green
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area, which is most likely the field
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Create a black image with the same dimensions as the input image
    result = np.zeros_like(image_path)

    # Draw the largest contour (the field) onto the black image
    cv2.drawContours(result, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the result image back to BGR color space
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Threshold the result image to create a binary mask
    _, mask = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    # Apply the mask to the input image to keep only the field, players, ball, and referee
    final_image = cv2.bitwise_and(image_path, image_path, mask=mask)

    return final_image


def run_inference(frame):
    # Remove the background
    frame = remove_background(frame)
    # Resize and pad image
    frame = letterbox(frame, 640, stride=64, auto=True)[0] # shape: (480, 640, 3)
    # Apply transforms
    frame = transforms.ToTensor()(frame) # torch.Size([3, 480, 640])
    # Convert image to float32
    frame = frame.to(torch.float32)
    # Match tensor type with model
    frame = frame.to(device)
    # Turn image into batch
    frame = frame.unsqueeze(0) # torch.Size([1, 3, 480, 640])
    output = model(frame)
    return output, frame

def setup(output, frame):
    inf_out = output['test']
    attn = output['attn']
    bases = output['bases']
    sem_output = output['sem']

    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = frame.shape
    names = model.names
    pooler_scale = model.pooler_scale

    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
                    
    # output, output_mask, output_mask_score, output_ac, output_ab
    output, output_mask, _, _, _ = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None) 


    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])

    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])

    pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)
                                                        
    # Detach Tensors from the device, send to the CPU and turn into NumPy arrays
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = frame[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
    return nimg, pred_masks_np, pred_cls, pred_conf


def extract_jersey_colors(original_image, pred_masks_np, pred_cls, pred_conf):
    jersey_features = []
    for one_mask, cls, conf in zip(pred_masks_np, pred_cls, pred_conf):
        if cls == 0 and conf >= 0.25:
            # Extract the color of the player's jersey
            one_mask_resized = cv2.resize(one_mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))
            jersey_pixels = original_image[one_mask_resized.astype(bool)]
            jersey_pixels = rgb2lab(jersey_pixels) # convert the image to LAB space
            
            # Compute the histogram of the jersey colors
            bins = np.arange(-128, 129, 16)
            hist, _ = np.histogramdd(jersey_pixels, bins=[bins, bins, bins])
            hist = hist.flatten()
            
            jersey_features.append(hist)
    return np.array(jersey_features)


def assign_jersey_labels(original_image, pred_img, pred_masks_np, pred_cls, pred_conf, plot_labels=True, label_offset=10):
    jersey_features = extract_jersey_colors(original_image, pred_masks_np, pred_cls, pred_conf)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(jersey_features)
    jersey_labels = kmeans.labels_
    player_masks = []
    ball_mask = None
    players_positions = []

    unique_labels = np.unique(jersey_labels)
    if len(unique_labels) > 2:
        # More than 2 teams, don't assign labels
        return pred_img, player_masks, ball_mask, players_positions

    for one_mask, cls, conf, label in zip(pred_masks_np, pred_cls, pred_conf, jersey_labels):
        if cls == 0 and conf >= 0.25:
            # Assign the jersey label to the player
            if label == 0:
                color = [255, 0, 0]  # Red for Team 1
            else:
                color = [0, 0, 255]  # Blue for Team 2
            pred_img[one_mask] = pred_img[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
            if plot_labels:
                y_center, x_center = np.mean(np.where(one_mask), axis=1)
                team_label = 'Team %d' % (label + 1)
                t_size = cv2.getTextSize(team_label, 0, fontScale=0.1, thickness=1)[0]
                label_pos = (int(x_center - t_size[0] / 2), int(y_center + t_size[1] / 2))
                pred_img = cv2.putText(pred_img, team_label, label_pos, 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                center_pos = (int(x_center), int(y_center - t_size[1] / 2 - label_offset))
                center_label = '(%d,%d)' % center_pos
                pred_img = cv2.putText(pred_img, center_label, center_pos, 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                players_positions.append((center_pos, label))  # add player position and label to the list
            player_masks.append(one_mask)
        elif cls == 32 and conf >= 0.25:
            pred_img[one_mask] = pred_img[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
            if plot_labels:
                y_center, x_center = np.mean(np.where(one_mask), axis=1)
                t_size = cv2.getTextSize("Ball", 0, fontScale=0.1, thickness=1)[0]
                label_pos = (int(x_center - t_size[0] / 2), int(y_center + t_size[1] / 2))
                pred_img = cv2.putText(pred_img, "Ball", label_pos, 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                center_pos = (int(x_center), int(y_center - t_size[1] / 2 - label_offset))
                players_positions.append((center_pos, -1))  # add ball position with label -1
                center_label = '(%d,%d)' % center_pos
                pred_img = cv2.putText(pred_img, center_label, center_pos, 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            ball_mask = one_mask

    # Split the players_positions list into positions for Team 1 and Team 2
    team1_positions = [pos for pos, label in players_positions if label == 0]
    team2_positions = [pos for pos, label in players_positions if label == 1]

    return pred_img, player_masks, ball_mask, team1_positions, team2_positions


def plot_results(original_image, plot_labels=True,selection_team="1"):
    # Resize the original image to match the size of the predicted image
    output, frame = run_inference(original_image)
    pred_img, pred_masks_np, pred_cls, pred_conf = setup(output, frame)
    original_image_resized = cv2.resize(original_image, (pred_img.shape[1], pred_img.shape[0]))
    # Assign jersey labels to each player
    pred_img, player_masks, ball_mask, team1_positions, team2_positions = assign_jersey_labels(original_image_resized, pred_img, pred_masks_np, pred_cls, pred_conf, plot_labels=plot_labels)    
    # Display the image
    if plot_labels:
        # Team 2 attack here
        # Get the farthest positions in each team
        # lw Selection f el parameter = 1 yeb2a team 2 hwa el attack
        # else Team 1 hwa el attack
        farthest_team1 = max(team1_positions)
        farthest_team2 = max(team2_positions)
        if selection_team == "1":
            attacking_team = "Team 1"
        elif selection_team == "2":
            attacking_team = "Team 2"
            
        # Get the last position of the attacking team
        if attacking_team == "Team 1":
            last_position = max(team1_positions)
        else:
            last_position = max(team2_positions)

        # Check for offside
        if last_position >= farthest_team2:
            text = "Offside!"
        else:
            text = "No offside."

        # Put the text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255) if text == "Offside!" else (0, 255, 0)
        thickness = 2
        org = (10, 30)
        cv2.putText(pred_img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        # Save the labeled image
        colored = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        # Add the prefix and suffix to the file name
        #file_name = os.path.basename(args.image_path)
        #new_file_name = "Decision_" + file_name
        #cv2.imwrite(new_file_name, colored)
        
        # Print the positions of each player
        print("Team 1:", sorted(team1_positions),'\n<br><br>')
        print("Team 2:", sorted(team2_positions),'\n<br><br>')
        print("Attack Team is: ",attacking_team)
    return pred_img


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to input image')
parser.add_argument('--video_path', type=str, help='Path to input video')
parser.add_argument('--Attack', type=str, help='Select attacking team 1,2')
args = parser.parse_args()

if args.image_path:
    img = cv2.imread(args.image_path)
    if(args.Attack == "1"):
        pred_img = plot_results(img, plot_labels=True,selection_team="1")
        colored = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        file_name = os.path.basename(args.image_path)
        new_file_name = "output/" + file_name
        cv2.imwrite(new_file_name, colored)
    if(args.Attack == "2"):
        pred_img = plot_results(img, plot_labels=True,selection_team="2")
        colored = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        file_name = os.path.basename(args.image_path)
        new_file_name = "output/" + file_name
        cv2.imwrite(new_file_name, colored)    
elif args.video_path:
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = 'output.mp4'  # replace with your desired output directory and filename
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        print("========================================================")
        print(f'Processing frame {current_frame}/{total_frames}')
        predicated_frame = plot_results(frame, plot_labels=True)  # pass frame to plot_results()
        colored_frame = cv2.cvtColor(predicated_frame, cv2.COLOR_BGR2RGB)
        out.write(colored_frame)  # write the processed frame to the output video
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()  # remember to release the output video
else:
    print('Please provide either an image or a video path.')
