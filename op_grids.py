import os
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import numpy as np
import cv2
import math

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

class GridMapper:
    def __init__(self, grid_size=100):
        self.grid = OccupancyGrid(size=grid_size)
        self.camera = CameraHandler()
        
        # Robot state
        self.robot_pos = (grid_size // 2, grid_size // 2)  # Center of grid
        self.robot_orientation = 0  # Degrees, 0 is facing positive X
        
        # Create windows
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Occupancy Grid', 800, 800)
    
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
    
    def run(self):
        """Main loop for mapping"""
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                # Handle keyboard input for robot movement
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):  # Rotate left
                    self.robot_orientation = (self.robot_orientation - 5) % 360
                elif key == ord('d'):  # Rotate right
                    self.robot_orientation = (self.robot_orientation + 5) % 360
                
                self.update_grid_from_camera(frame)
                grid_vis = self.grid.visualize()
                
                # Show both camera feed and grid
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