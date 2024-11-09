import os
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
import numpy as np
import cv2
import math
import torch
from torchvision import transforms
from PIL import Image

class OccupancyGrid:
    def __init__(self, size=100):
        """
        Initialize occupancy grid
        Args:
            size (int): Size of grid (size x size)
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.center = (size // 2, size // 2)
        self.cell_size = 0.5  # Each cell represents 0.5 meters
        
        # Define cell states
        self.UNKNOWN = 0
        self.FREE = 1
        self.OCCUPIED = 2
        
        # Define colors for visualization (BGR format)
        self.colors = {
            self.UNKNOWN: (128, 128, 128),  # Gray
            self.FREE: (0, 255, 0),         # Green
            self.OCCUPIED: (0, 0, 255)      # Red
        }
        
        # Initialize grid with unknown state
        self.initialize_grid()

    def initialize_grid(self):
        """Set all cells to unknown state"""
        self.grid.fill(self.UNKNOWN)
    
    def update_cell(self, x, y, state):
        """
        Update the state of a single cell
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            state (int): Cell state (UNKNOWN, FREE, or OCCUPIED)
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[y, x] = state
    
    def get_cell(self, x, y):
        """Get the state of a cell"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[y, x]
        return None
    
    def visualize(self):
        """
        Create visualization of the occupancy grid
        Returns:
            numpy.ndarray: RGB image of the grid
        """
        # Create RGB visualization array
        vis_grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Color cells based on state
        for state, color in self.colors.items():
            vis_grid[self.grid == state] = color
            
        # Add grid lines every 10 cells (10cm)
        for i in range(0, self.size, 10):
            vis_grid[i, :] = [255, 255, 255]  # White horizontal lines
            vis_grid[:, i] = [255, 255, 255]  # White vertical lines
            
        # Mark center/robot position
        cv2.circle(vis_grid, self.center, 3, (255, 255, 0), -1)  # Yellow dot
        
        return vis_grid

    def meters_to_cells(self, meters_x, meters_y):
        """Convert real-world meters to grid cell coordinates"""
        cell_x = int(meters_x / self.cell_size) + self.center[0]
        cell_y = int(meters_y / self.cell_size) + self.center[1]
        return cell_x, cell_y

    def cells_to_meters(self, cell_x, cell_y):
        """Convert grid cell coordinates to real-world meters"""
        meters_x = (cell_x - self.center[0]) * self.cell_size
        meters_y = (cell_y - self.center[1]) * self.cell_size
        return meters_x, meters_y

class CameraHandler:
    def __init__(self):
        """Initialize camera and set properties"""
        # Try different camera indices if the default doesn't work
        for i in range(2):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
        
        # Camera settings
        if self.cap.isOpened():
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Try to set backend-specific properties to avoid warning
            # This is macOS specific
            try:
                # Try to use regular webcam first
                self.cap.set(cv2.CAP_PROP_PVAPI_PIXELFORMAT, 0)
            except:
                pass
        else:
            raise RuntimeError("Failed to open camera")
        
        # Rest of your initialization code remains the same
        self.fov_horizontal = 60
        self.fov_vertical = 45
        self.min_object_area = 500
        self.blur_size = (5, 5)

    def get_frame(self):
        """Capture and return a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def detect_objects(self, frame):
        """
        Detect objects in the frame using improved methods
        Returns: List of contours that likely represent objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_size, 0)
        
        # Use adaptive thresholding instead of simple threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on area
        return [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_object_area]

    def release(self):
        """Release the camera"""
        self.cap.release()

class DepthEstimator:
    def __init__(self):
        """Initialize depth estimation models and parameters"""
        # Initialize MiDaS model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()
        
        # Modified transform to ensure consistent size
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # For traditional approach
        self.feature_detector = cv2.SIFT_create()
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None

    def estimate_depth_ai(self, frame):
        """Estimate depth using MiDaS"""
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Transform image to fixed size
        input_batch = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize back to original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame.shape[0], frame.shape[1]),  # Original frame size
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype(np.uint8)
        
        return depth_map

    def estimate_depth_traditional(self, frame):
        """Estimate depth using traditional computer vision"""
        if self.previous_frame is None:
            self.previous_frame = frame
            self.previous_keypoints, self.previous_descriptors = \
                self.feature_detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
            return None

        # Detect features in current frame
        current_keypoints, current_descriptors = \
            self.feature_detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

        # Match features
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(self.previous_descriptors, current_descriptors, k=2)

        # Calculate relative depths from motion
        depth_map = np.zeros(frame.shape[:2], dtype=np.float32)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                pt1 = self.previous_keypoints[m.queryIdx].pt
                pt2 = current_keypoints[m.trainIdx].pt
                # Estimate relative depth from motion magnitude
                motion = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                depth_map[int(pt2[1]), int(pt2[0])] = 1.0 / (motion + 1e-5)

        self.previous_frame = frame
        self.previous_keypoints = current_keypoints
        self.previous_descriptors = current_descriptors
        
        # Normalize and smooth depth map
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
        depth_map = (depth_map * 255).astype(np.uint8)
        
        return depth_map

class GridMapper:
    def __init__(self, grid_size=100):
        self.grid = OccupancyGrid(size=grid_size)
        self.camera = CameraHandler()
        self.depth_estimator = DepthEstimator()
        
        # Robot state
        self.robot_pos = (grid_size // 2, grid_size // 2)
        self.robot_orientation = 0
        
        # Create windows
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Occupancy Grid', 800, 800)
        
        # Depth estimation mode
        self.use_ai_depth = True  # Toggle between AI and traditional methods
    
    def calculate_fov_cells(self):
        """Calculate grid cells that fall within camera FOV"""
        fov_cells = []
        
        # Convert FOV and orientation to radians
        fov_rad = math.radians(self.camera.fov_horizontal)
        orientation_rad = math.radians(self.robot_orientation)
        
        # Calculate FOV lines
        left_angle = orientation_rad - fov_rad/2
        right_angle = orientation_rad + fov_rad/2
        
        # Distance to check (half of grid size)
        max_dist = self.grid.size // 2
        
        # Generate points along FOV lines
        for r in range(max_dist):
            # Calculate arc at this radius
            arc_length = int(r * math.tan(fov_rad/2)) + 1
            
            for i in range(-arc_length, arc_length + 1):
                # Calculate x, y coordinates
                x = int(self.robot_pos[0] + r * math.cos(orientation_rad) + 
                       i * math.cos(orientation_rad + math.pi/2))
                y = int(self.robot_pos[1] + r * math.sin(orientation_rad) + 
                       i * math.sin(orientation_rad + math.pi/2))
                
                if 0 <= x < self.grid.size and 0 <= y < self.grid.size:
                    fov_cells.append((x, y))
        
        return fov_cells
    
    def update_grid_from_camera(self, frame):
        """Update grid based on camera frame"""
        # Get cells in FOV
        fov_cells = self.calculate_fov_cells()
        
        # Mark FOV cells as free initially
        for x, y in fov_cells:
            self.grid.update_cell(x, y, self.grid.FREE)
        
        # Detect objects
        objects = self.camera.detect_objects(frame)
        
        # Draw detected objects on camera feed
        cv2.drawContours(frame, objects, -1, (0, 255, 0), 2)
        
        # Update grid with detected objects
        for contour in objects:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Improved coordinate transformation
                # This is still simplified - you might want to add proper camera calibration
                grid_x = self.robot_pos[0] + (cx - frame.shape[1]//2) // 6
                grid_y = self.robot_pos[1] + (cy - frame.shape[0]//2) // 6
                
                # Mark as occupied and surrounding cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        self.grid.update_cell(
                            grid_x + dx, 
                            grid_y + dy, 
                            self.grid.OCCUPIED
                        )
    
    def update_grid_from_depth(self, depth_map, frame_width, frame_height):
        """Update grid using depth information"""
        # Camera parameters (should be calibrated for your camera)
        focal_length = frame_width / (2 * math.tan(math.radians(self.camera.fov_horizontal / 2)))
        
        for y in range(frame_height):
            for x in range(frame_width):
                depth = depth_map[y, x]
                if depth > 10:  # Threshold to filter out noise
                    # Convert pixel coordinates to world coordinates
                    world_x = (x - frame_width/2) * depth / focal_length
                    world_y = (y - frame_height/2) * depth / focal_length
                    
                    # Convert world coordinates (in mm) to meters
                    world_x_m = world_x / 1000.0
                    world_y_m = world_y / 1000.0
                    
                    # Convert to grid coordinates
                    grid_x, grid_y = self.grid.meters_to_cells(world_x_m, world_y_m)
                    
                    if 0 <= grid_x < self.grid.size and 0 <= grid_y < self.grid.size:
                        self.grid.update_cell(grid_x, grid_y, self.grid.OCCUPIED)
                        
                        # Mark cells between camera and object as free
                        self.mark_free_path(grid_x, grid_y)

    def mark_free_path(self, target_x, target_y):
        """Mark cells between camera and target as free using Bresenham's line algorithm"""
        x0, y0 = self.robot_pos
        x1, y1 = target_x, target_y
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x != x1 or y != y1:
            if 0 <= x < self.grid.size and 0 <= y < self.grid.size:
                self.grid.update_cell(x, y, self.grid.FREE)
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def run(self):
        """Main loop for mapping"""
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Toggle depth estimation mode
                    self.use_ai_depth = not self.use_ai_depth
                elif key == ord('a'):
                    self.robot_orientation = (self.robot_orientation - 5) % 360
                elif key == ord('d'):
                    self.robot_orientation = (self.robot_orientation + 5) % 360
                
                # Estimate depth
                if self.use_ai_depth:
                    depth_map = self.depth_estimator.estimate_depth_ai(frame)
                else:
                    depth_map = self.depth_estimator.estimate_depth_traditional(frame)
                
                if depth_map is not None:
                    # Update grid using depth information
                    self.update_grid_from_depth(depth_map, frame.shape[1], frame.shape[0])
                    
                    # Show depth map
                    cv2.imshow('Depth Map', depth_map)
                
                # Update and show visualizations
                self.update_grid_from_camera(frame)
                grid_vis = self.grid.visualize()
                
                cv2.imshow('Camera Feed', frame)
                cv2.imshow('Occupancy Grid', grid_vis)
                
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

def main():
    """Main function to run the grid mapper"""
    mapper = GridMapper()
    mapper.run()

if __name__ == "__main__":
    main() 