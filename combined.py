import cv2
import torch
import time
import tkinter as tk
from tkinter import Label, StringVar
import warnings
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, message="torch.cuda.amp.autocast(args...) is deprecated. Please use torch.amp.autocast('cuda', args...) instead.")

# Load the YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

# Load the background image
background_image = cv2.imread('lanes.jpg')
image_height, image_width = background_image.shape[:2]

# Video paths for four lanes (assuming separate videos for each lane)
video_paths = [
    'input/video/lane1.mp4',
    'input/video/lane2.mp4',
    'input/video/lane3.mp4',
    'input/video/lane4.mp4',
]

# Initialize video captures for each lane
caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
print("Video captures initialized.")

# Check if all videos were opened successfully
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video for Lane {idx + 1}.")
        exit()  # Exit if any video could not be opened

# Initialize video writers for each lane
output_video_dir = 'output/video/'
os.makedirs(output_video_dir, exist_ok=True)

fps = int(caps[0].get(cv2.CAP_PROP_FPS))
frame_size = (int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec

video_writers = []
for i in range(4):
    output_path = os.path.join(output_video_dir, f'lane{i+1}_processed.avi')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    video_writers.append(video_writer)
print("Video writers initialized.")

# Initialize baseline green light time (in seconds)
baseline_time = 1  # 1 second for each lane

# Initialize last active times for each lane
last_active_times = [0] * 4

# Initialize the tkinter window
root = tk.Tk()
root.title("Traffic Signal Simulation")

# Variables to store lane states and vehicle counts
lane_states = [StringVar() for _ in range(4)]
vehicle_counts_vars = [StringVar() for _ in range(4)]

# Initialize labels for lane states and vehicle counts
for i in range(4):
    Label(root, text=f"Lane {i+1} Vehicle Count:").grid(row=i, column=0, padx=10, pady=10)
    Label(root, textvariable=vehicle_counts_vars[i]).grid(row=i, column=1, padx=10, pady=10)
    Label(root, text=f"Lane {i+1} Signal:").grid(row=i, column=2, padx=10, pady=10)
    Label(root, textvariable=lane_states[i]).grid(row=i, column=3, padx=10, pady=10)
    vehicle_counts_vars[i].set("0")
    lane_states[i].set("Red")

# Initialize CSV logging
log_file_path = 'output/traffic_log.csv'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

with open(log_file_path, 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Cycle', 'Lane', 'Vehicle Count', 'Signal', 'Green Light Time'])

# Function to count vehicles in a frame using YOLOv5
def count_vehicles(frame):
    results = model(frame)
    vehicle_count = len(results.pandas().xyxy[0])
    return vehicle_count

# Function to create and update the simulation window
def update_simulation_window(lane_statuses, lane_timings, current_lane, time_left, last_active_times):
    # Use the background image as the base for the simulation window
    window = np.zeros((1124, 1920, 3), dtype=np.uint8)  # Full HD window size
    window[:image_height, :image_width] = background_image.copy()

    lane_names = ["North (Lane 1)", "East (Lane 2)", "South (Lane 3)", "West (Lane 4)"]
    colors = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255)}

    # Define positions for each lane's text information (adjusted)
    positions = [
        (1000, 100),  # North (Lane 1)
        (1000, 300),  # East (Lane 2)
        (1000, 500),  # South (Lane 3)
        (1000, 700),  # West (Lane 4)
    ]

    # Define positions for traffic lights (small circles)
    light_positions = [
        (310, 260),  # North (Lane 1)
        (310, 535),  # East (Lane 2)
        (540, 260),  # South (Lane 3)
        (545, 535),  # West (Lane 4)
    ]

    # Draw the lane status, timing information, and last active times
    for i in range(4):
        lane_color = colors["Green"] if i == current_lane else colors["Red"]
        cv2.putText(window, lane_names[i], positions[i], cv2.FONT_HERSHEY_SIMPLEX, 1.5, lane_color, 2, cv2.LINE_AA)
        cv2.putText(window, f"Status: {lane_statuses[i]}", (positions[i][0], positions[i][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, lane_color, 2, cv2.LINE_AA)
        cv2.putText(window, f"Time Left: {time_left:.2f} sec" if i == current_lane else f"Next in: {lane_timings[i]:.2f} sec", 
                    (positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors["Yellow"], 2, cv2.LINE_AA)
        cv2.putText(window, f"Last Active: {last_active_times[i]:.2f} sec ago", 
                    (positions[i][0], positions[i][1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors["Yellow"], 2, cv2.LINE_AA)

        # Draw the traffic lights (small circles)
        # Green light
        green_light_color = colors["Green"] if i == current_lane else (0, 100, 0)
        cv2.circle(window, light_positions[i], 20, green_light_color, -1)

        # Red light (slightly below the green light)
        red_light_color = (100, 0, 0) if i == current_lane else colors["Red"]
        cv2.circle(window, (light_positions[i][0], light_positions[i][1] + 50), 20, red_light_color, -1)

    # Show the simulation window
    cv2.imshow("Traffic Signal Simulation", window)

# Function to process a frame for a single lane
def process_lane(idx, cap, video_writer):
    ret, frame = cap.read()
    if not ret:
        print(f"Video for Lane {idx + 1} has ended.")
        return None, None  # Indicate that the video has ended

    print(f"Frame captured from Lane {idx + 1}.")

    # Resize frame if not matching the expected size
    if frame.shape[1] != image_width or frame.shape[0] != image_height:
        frame = cv2.resize(frame, (image_width, image_height))
        print(f"Resized frame for Lane {idx + 1}.")

    # Get vehicle count and detection results
    vehicle_count = count_vehicles(frame)
    
    # Draw bounding boxes on the frame
    results = model(frame)
    for *xyxy, conf, cls in results.xyxy[0]:
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    
    # Write the processed frame to output video
    video_writer.write(frame)

    return vehicle_count, frame

# Main loop to simulate the traffic signal operation
def update_gui():
    global cycle_count
    all_videos_ended = True
    print("\nProcessing frames from all lanes...")
    vehicle_counts = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_lane, idx, caps[idx], video_writers[idx]) for idx in range(4)]
        results = [f.result() for f in futures]

    if len(results) != 4:
        print("Error: Not all lanes provided frames, stopping simulation.")
        return

    vehicle_counts = [result[0] for result in results if result[0] is not None]

    # Calculate the current cycle time
    current_time = time.time()
    elapsed_time = current_time - start_time
    cycle_time = elapsed_time % (4 * baseline_time)  # Total cycle time (4 lanes * baseline_time)
    current_cycle = int(cycle_time / baseline_time) % 4
    time_left = baseline_time - (elapsed_time % baseline_time)

    # Update lane statuses and vehicle counts
    lane_statuses = ["Red"] * 4
    lane_timings = [baseline_time] * 4

    # Determine the current active lane
    lane_statuses[current_cycle] = "Green"
    lane_timings[current_cycle] = time_left

    # Update GUI
    for i in range(4):
        vehicle_counts_vars[i].set(str(vehicle_counts[i]))
        lane_states[i].set(lane_statuses[i])

    # Update the simulation window
    update_simulation_window(lane_statuses, lane_timings, current_cycle, time_left, last_active_times)
    cv2.waitKey(1)  # Allow OpenCV to process the window update

    # Log the data
    with open(log_file_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        for i in range(4):
            log_writer.writerow([cycle_count, i + 1, vehicle_counts[i], lane_statuses[i], baseline_time - time_left])

    # Check if all videos have ended
    all_videos_ended = all(not cap.isOpened() for cap in caps)

    if all_videos_ended:
        print("All videos have ended. Stopping simulation.")
        root.quit()
        cv2.destroyAllWindows()
        for cap in caps:
            cap.release()
        for video_writer in video_writers:
            video_writer.release()
        exit()

    # Increment the cycle count
    cycle_count += 1

# Initialize the GUI and start the simulation loop
cycle_count = 0
start_time = time.time()
while True:
    update_gui()
    root.update_idletasks()
    root.update()
