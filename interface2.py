import cv2
import torch
import time
import numpy as np

# Load the YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

# Load the background image
background_image = cv2.imread('lanes.jpg')

# Get the dimensions of the background image
image_height, image_width = background_image.shape[:2]

# Video paths for four lanes (assuming separate videos for each lane)
video_paths = [
    'input/video/lane1.mp4',  # Lane 1 (e.g., North)
    'input/video/lane2.mp4',  # Lane 2 (e.g., East)
    'input/video/lane3.mp4',  # Lane 3 (e.g., South)
    'input/video/lane4.mp4',  # Lane 4 (e.g., West)
]

# Initialize video captures for each lane
caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
print("Video captures initialized.")

# Check if all videos were opened successfully
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video for Lane {idx + 1}.")
        exit()  # Exit if any video could not be opened

# Initialize baseline green light time (in seconds)
baseline_time = 1  # 30 seconds for each lane

# Initialize last active times for each lane
last_active_times = [0] * 4

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
    colors = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255), "TextBackground": (50, 50, 50)}

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
        
        # Draw rectangle backgrounds for text
        cv2.rectangle(window, (positions[i][0] - 10, positions[i][1] - 30), 
                      (positions[i][0] + 500, positions[i][1] + 150), colors["TextBackground"], -1)

        # Draw lane names and status information
        cv2.putText(window, lane_names[i], positions[i], cv2.FONT_HERSHEY_SIMPLEX, 1.5, lane_color, 2, cv2.LINE_AA)
        cv2.putText(window, f"Status: {lane_statuses[i]}", (positions[i][0], positions[i][1] + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, lane_color, 2, cv2.LINE_AA)
        cv2.putText(window, f"Time Left: {time_left:.2f} sec" if i == current_lane else f"Next in: {lane_timings[i]:.2f} sec", 
                    (positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors["Yellow"], 2, cv2.LINE_AA)
        cv2.putText(window, f"Last Active: {last_active_times[i]:.2f} sec ago", 
                    (positions[i][0], positions[i][1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors["Yellow"], 2, cv2.LINE_AA)

        # Draw the traffic lights (small circles)
        green_light_color = colors["Green"] if i == current_lane else (0, 100, 0)
        cv2.circle(window, light_positions[i], 20, green_light_color, -1)

        red_light_color = (100, 0, 0) if i == current_lane else colors["Red"]
        cv2.circle(window, (light_positions[i][0], light_positions[i][1] + 50), 20, red_light_color, -1)

    # Show the simulation window
    cv2.imshow("Traffic Signal Simulation", window)

# Main loop to simulate the traffic signal operation
start_time = time.time()

while all(cap.isOpened() for cap in caps):
    vehicle_counts = []

    # Capture and process frames for each lane
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from Lane {idx + 1}.")
            break
        vehicle_count = count_vehicles(frame)
        vehicle_counts.append(vehicle_count)

    # Ensure processing continues only if all frames were read successfully
    if len(vehicle_counts) != 4:
        print("Error: Not all lanes provided frames, stopping simulation.")
        break

    # Calculate total vehicles across all lanes
    total_vehicles = sum(vehicle_counts) if sum(vehicle_counts) != 0 else 1

    # Determine green light time for each lane based on vehicle count
    lane_timings = [
        baseline_time + (count / total_vehicles) * 30
        for count in vehicle_counts
    ]

    # Simulate the signal operation in sequence
    for i in range(4):
        lane_statuses = ["Green" if i == j else "Red" for j in range(4)]
        last_active_times[i] = time.time() - start_time

        time_left = lane_timings[i]
        while time_left > 0:
            update_simulation_window(lane_statuses, lane_timings, i, time_left, last_active_times)
            
            time.sleep(1)
            time_left -= 1
            last_active_times = [time.time() - start_time if j != i else 0 for j in range(4)]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release all captures and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print("All video captures released, program ended.")
