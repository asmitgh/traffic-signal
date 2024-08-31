import cv2
import torch
import time
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
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

# Initialize the tkinter window with ttkbootstrap for modern styling
root = ttk.Window(themename="cyborg")  # Choose a dark aesthetic theme like "cyborg"
root.title("Traffic Signal Simulation")
root.geometry("600x400")

# Variables to store lane states and vehicle counts
lane_states = [ttk.StringVar() for _ in range(4)]
vehicle_counts_vars = [ttk.StringVar() for _ in range(4)]

# Frame for the lane status display
status_frame = ttk.Frame(root, padding=(20, 10), style="primary.TFrame")
status_frame.grid(row=0, column=0, sticky=NSEW)

# Initialize labels for lane states and vehicle counts
for i in range(4):
    ttk.Label(status_frame, text=f"Lane {i+1} Vehicle Count:", font=("Helvetica", 12, "bold"), foreground="white").grid(row=i, column=0, padx=10, pady=10)
    ttk.Label(status_frame, textvariable=vehicle_counts_vars[i], font=("Helvetica", 12), foreground="lightgreen").grid(row=i, column=1, padx=10, pady=10)
    ttk.Label(status_frame, text=f"Lane {i+1} Signal:", font=("Helvetica", 12, "bold"), foreground="white").grid(row=i, column=2, padx=10, pady=10)
    ttk.Label(status_frame, textvariable=lane_states[i], font=("Helvetica", 12), foreground="red").grid(row=i, column=3, padx=10, pady=10)
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
    cv2.imshow('Traffic Signal Simulation', window)
    cv2.waitKey(1)  # Wait for a short moment (1 ms) to allow the window to refresh

# Function to update vehicle counts and lane statuses
def update_status(cycle, lane, count, signal, green_time):
    lane_states[lane].set(signal)
    vehicle_counts_vars[lane].set(str(count))

    # Log data to CSV file
    with open(log_file_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([cycle, lane + 1, count, signal, green_time])

# Function to process and simulate traffic management
def process_lanes():
    cycle = 1  # Initialize the simulation cycle

    while True:
        with ThreadPoolExecutor(max_workers=4) as executor:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video for Lane {i + 1}")
                    return

                frames.append(frame)

            # Concurrently process vehicle counts for each lane
            futures = [executor.submit(count_vehicles, frame) for frame in frames]
            vehicle_counts = [future.result() for future in futures]

        total_count = sum(vehicle_counts)
        green_light_durations = [baseline_time + (count / total_count) * baseline_time * 5 if total_count > 0 else baseline_time for count in vehicle_counts]

        current_lane = cycle % 4

        update_status(cycle, current_lane, vehicle_counts[current_lane], "Green", green_light_durations[current_lane])
        print(f"Cycle {cycle}: Lane {current_lane + 1} Green Light - {green_light_durations[current_lane]:.2f} sec")

        start_time = time.time()

        while time.time() - start_time < green_light_durations[current_lane]:
            update_simulation_window(["Green" if i == current_lane else "Red" for i in range(4)], green_light_durations, current_lane, green_light_durations[current_lane] - (time.time() - start_time), last_active_times)

        last_active_times[current_lane] = 0
        for i in range(4):
            if i != current_lane:
                last_active_times[i] += green_light_durations[current_lane]

        lane_states[current_lane].set("Red")
        vehicle_counts_vars[current_lane].set("0")

        cycle += 1

# Start the traffic signal simulation
process_lanes()

# Start the Tkinter main loop
root.mainloop()

# Release all resources
cv2.destroyAllWindows()
for cap, video_writer in zip(caps, video_writers):
    cap.release()
    video_writer.release()
