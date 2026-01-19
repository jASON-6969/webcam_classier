"""
Image display utility module
Supports loading and displaying multiple image formats
"""
import os
from PIL import Image, ImageTk
import tkinter as tk


class ImageDisplay:
    """Image display manager"""
    
    def __init__(self, base_path="to_do/image"):
        """
        Initialize image display manager
        
        Args:
            base_path: Base path for images
        """
        self.base_path = base_path
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.avif', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
        
        # Image path mapping (auto-scan directories)
        self.image_paths = {}
        
        # GIF animation management
        self.gif_frames = {}  # Store all frames for each GIF: {class_label: [frame1, frame2, ...]}
        self.gif_delays = {}  # Store delay time for each frame (milliseconds): {class_label: [delay1, delay2, ...]}
        self.animation_jobs = {}  # Store animation task IDs: {class_label: after_id}
        self.animation_labels = {}  # Store Label widgets playing animations: {class_label: tk.Label}
        self.animation_indices = {}  # Store current frame index: {class_label: current_index}
        
        # Automatically scan image files in directories
        self._scan_image_directories()
        
        # Verify image files exist
        self._verify_images()
    
    def _label_to_directory(self, class_label):
        """
        Convert class label to directory name
        
        Args:
            class_label: Class label (e.g., 'Class 1' or 'Class 2')
            
        Returns:
            Directory name (e.g., 'class1' or 'calss2')
        """
        # Convert "Class 1" to "class1", "Class 2" to "calss2", etc.
        # Note: Class 2 corresponds to directory name "calss2" (typo but needs to maintain compatibility)
        label_lower = class_label.lower().replace(' ', '')
        if label_lower == 'class2':
            return 'calss2'  # Special handling for typo
        return label_lower
    
    def _find_first_image_in_directory(self, directory_path):
        """
        Find the first supported image file in the specified directory
        
        Args:
            directory_path: Directory path
            
        Returns:
            Found image file path, or None if not found
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            return None
        
        # Get all files in directory, sorted by filename
        try:
            files = sorted(os.listdir(directory_path))
            for filename in files:
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    # Check file extension
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in self.supported_formats:
                        return file_path
        except Exception as e:
            print(f"Error scanning directory {directory_path}: {e}")
        
        return None
    
    def _scan_image_directories(self):
        """Automatically scan directories under base path to find image files"""
        if not os.path.exists(self.base_path):
            print(f"Warning: Base path does not exist: {self.base_path}")
            return
        
        # Scan all subdirectories under base path
        try:
            for item in os.listdir(self.base_path):
                item_path = os.path.join(self.base_path, item)
                if os.path.isdir(item_path):
                    # Try to convert directory name to label name
                    # Example: class1 -> Class 1, calss2 -> Class 2
                    class_label = self._directory_to_label(item)
                    if class_label:
                        # Find first image file in directory
                        image_path = self._find_first_image_in_directory(item_path)
                        if image_path:
                            self.image_paths[class_label] = image_path
                            print(f"Auto-discovered: {class_label} -> {image_path}")
                        else:
                            print(f"Warning: No supported image files found in directory {item_path}")
        except Exception as e:
            print(f"Error scanning directories: {e}")
    
    def _directory_to_label(self, directory_name):
        """
        Convert directory name to class label
        
        Args:
            directory_name: Directory name (e.g., 'class1' or 'calss2')
            
        Returns:
            Class label (e.g., 'Class 1' or 'Class 2'), or None if cannot convert
        """
        # Handle special spelling: calss2 -> Class 2
        if directory_name.lower() == 'calss2':
            return 'Class 2'
        
        # Handle standard format: class1 -> Class 1, class2 -> Class 2
        if directory_name.lower().startswith('class'):
            try:
                # Extract number part
                number = directory_name.lower().replace('class', '').strip()
                if number.isdigit():
                    return f'Class {number}'
            except:
                pass
        
        return None
    
    def _verify_images(self):
        """Verify image files exist"""
        for label, path in self.image_paths.items():
            if os.path.exists(path):
                print(f"✓ Found image for {label}: {path}")
            else:
                print(f"✗ Warning: Image for {label} does not exist: {path}")
    
    def _load_gif_frames(self, image_path, max_width=400, max_height=400):
        """
        Load all frames of a GIF
        
        Args:
            image_path: GIF file path
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            (frames, delays) tuple, frames is list of processed frames, delays is list of delay times (milliseconds)
        """
        frames = []
        delays = []
        
        try:
            pil_image = Image.open(image_path)
            
            # Get GIF delay time (default 100 milliseconds)
            default_delay = 100
            if 'duration' in pil_image.info:
                default_delay = pil_image.info['duration']
            
            frame_count = 0
            while True:
                try:
                    # Copy current frame
                    frame = pil_image.copy()
                    
                    # Handle transparency and mode conversion
                    if frame.mode == 'RGBA':
                        background = Image.new('RGB', frame.size, (255, 255, 255))
                        if len(frame.split()) == 4:
                            background.paste(frame, mask=frame.split()[3])
                        else:
                            background.paste(frame)
                        frame = background
                    elif frame.mode == 'P':
                        if 'transparency' in pil_image.info:
                            frame = frame.convert('RGBA')
                            background = Image.new('RGB', frame.size, (255, 255, 255))
                            if len(frame.split()) > 3:
                                background.paste(frame, mask=frame.split()[3])
                            else:
                                background.paste(frame)
                            frame = background
                        else:
                            frame = frame.convert('RGB')
                    elif frame.mode not in ('RGB', 'L'):
                        frame = frame.convert('RGB')
                    
                    # Resize
                    width, height = frame.size
                    scale_width = max_width / width
                    scale_height = max_height / height
                    scale = min(scale_width, scale_height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Create fixed-size background image (white background)
                    final_image = Image.new('RGB', (max_width, max_height), color='white')
                    x_offset = (max_width - new_width) // 2
                    y_offset = (max_height - new_height) // 2
                    final_image.paste(frame, (x_offset, y_offset))
                    
                    frames.append(final_image)
                    
                    # Get current frame delay time
                    # PIL's duration unit is seconds, need to convert to milliseconds
                    delay = default_delay
                    try:
                        # Try to get delay time from current frame's info
                        frame_info = pil_image.info if hasattr(pil_image, 'info') else {}
                        if 'duration' in frame_info:
                            delay_seconds = frame_info['duration']
                            # Convert to milliseconds, use default if 0 or invalid
                            if delay_seconds and delay_seconds > 0:
                                delay = int(delay_seconds * 1000)  # seconds to milliseconds
                            else:
                                delay = default_delay
                        else:
                            delay = default_delay
                    except Exception as e:
                        print(f"Failed to get delay time for frame {frame_count}: {e}")
                        delay = default_delay
                    
                    # Ensure delay time is within reasonable range (50-2000ms, use 100ms if over 2000)
                    if delay > 2000:
                        delay = 100  # If delay too long, use default
                    delay = max(50, min(2000, int(delay)))
                    delays.append(delay)
                    if frame_count < 5:  # Only print debug info for first 5 frames
                        print(f"Debug: Frame {frame_count} delay time: {delay} ms")
                    
                    frame_count += 1
                    
                    # Try to read next frame
                    pil_image.seek(frame_count)
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error loading GIF frame {frame_count}: {e}")
                    break
            
            return frames, delays
            
        except Exception as e:
            print(f"Failed to load GIF frames: {e}")
            return [], []
    
    def load_image(self, class_label, max_width=400, max_height=400):
        """
        Load and resize image
        
        Args:
            class_label: Class label (e.g., 'Class 1' or 'Class 2')
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            ImageTk.PhotoImage object, or None if loading fails
        """
        try:
            # Get image path
            image_path = self.image_paths.get(class_label)
            if not image_path:
                print(f"Image path not found for class {class_label}")
                return None
            
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                return None
            
            # Use PIL to load image (supports multiple formats)
            pil_image = Image.open(image_path)
            
            # Handle GIF format
            if pil_image.format == 'GIF':
                # Check if it's a multi-frame GIF
                try:
                    pil_image.seek(1)  # Try to read second frame
                    is_animated = True
                    pil_image.seek(0)  # Return to first frame
                except EOFError:
                    is_animated = False
                    pil_image.seek(0)
                
                # If animated GIF, load all frames
                if is_animated:
                    print(f"Debug: Detected animated GIF: {class_label}")
                    frames, delays = self._load_gif_frames(image_path, max_width, max_height)
                    print(f"Debug: Loaded {len(frames)} frames, delay times: {delays[:5]}...")
                    if frames:
                        self.gif_frames[class_label] = frames
                        self.gif_delays[class_label] = delays
                        print(f"Loaded animated GIF: {class_label}, total {len(frames)} frames")
                        # Return first frame as initial display (already processed RGB image)
                        pil_image = frames[0]
                        # Directly convert to PhotoImage and return, skip subsequent processing
                        photo = ImageTk.PhotoImage(pil_image)
                        return photo
                    else:
                        print(f"Debug: GIF frame loading failed: {class_label}")
                        # If loading fails, use first frame
                        pil_image.seek(0)
                        is_animated = False
                else:
                    print(f"Debug: Single-frame GIF: {class_label}")
                    # Single-frame GIF, normal processing
                    pil_image.seek(0)
                
                # GIF is usually palette mode (P) or RGBA mode
                if not is_animated and pil_image.mode == 'P':
                    # Check if there's transparency
                    if 'transparency' in pil_image.info:
                        # Convert to RGBA to preserve transparency info
                        pil_image = pil_image.convert('RGBA')
                    else:
                        # No transparency, directly convert to RGB
                        pil_image = pil_image.convert('RGB')
            
            # Handle transparency and mode conversion (only for non-animated GIF or non-GIF images)
            if pil_image.mode == 'RGBA':
                # If there's transparency channel, create white background first
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'RGBA' and len(pil_image.split()) == 4:
                    background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
                else:
                    background.paste(pil_image)
                pil_image = background
            elif pil_image.mode == 'P':
                # Palette mode, convert to RGBA then RGB
                try:
                    pil_image = pil_image.convert('RGBA')
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    if len(pil_image.split()) > 3:
                        background.paste(pil_image, mask=pil_image.split()[3])
                    else:
                        background.paste(pil_image)
                    pil_image = background
                except Exception as e:
                    print(f"Palette mode conversion failed: {e}, trying direct conversion")
                    pil_image = pil_image.convert('RGB')
            elif pil_image.mode not in ('RGB', 'L'):
                # Convert other modes to RGB
                try:
                    pil_image = pil_image.convert('RGB')
                except Exception as e:
                    print(f"Mode conversion failed ({pil_image.mode}): {e}")
                    raise
            
            # Fixed output size, maintain aspect ratio and add background padding
            width, height = pil_image.size
            
            # Calculate scale factor to ensure image fits fixed size (maintain aspect ratio)
            scale_width = max_width / width
            scale_height = max_height / height
            scale = min(scale_width, scale_height)
            
            # Calculate scaled size
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image (maintain aspect ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create fixed-size background image (white background)
            final_image = Image.new('RGB', (max_width, max_height), color='white')
            
            # Calculate center position
            x_offset = (max_width - new_width) // 2
            y_offset = (max_height - new_height) // 2
            
            # Paste scaled image at center position
            final_image.paste(pil_image, (x_offset, y_offset))
            
            # Use final image
            pil_image = final_image
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            return photo
            
        except Exception as e:
            error_msg = f"Failed to load image ({class_label}): {e}"
            print(error_msg)
            print(f"Image path: {image_path}")
            print(f"File exists: {os.path.exists(image_path) if image_path else False}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_image_path(self, class_label):
        """
        Get image path for specified class
        
        Args:
            class_label: Class label
            
        Returns:
            Image path string, or None if not exists
        """
        return self.image_paths.get(class_label)
    
    def add_image_path(self, class_label, image_path):
        """
        Add new image path mapping
        
        Args:
            class_label: Class label
            image_path: Image path
        """
        self.image_paths[class_label] = image_path
        if os.path.exists(image_path):
            print(f"✓ Added image mapping: {class_label} -> {image_path}")
        else:
            print(f"✗ Warning: Image file does not exist: {image_path}")
    
    def start_animation(self, class_label, label_widget, root):
        """
        Start GIF animation
        
        Args:
            class_label: Class label
            label_widget: Label widget for displaying animation
            root: Tkinter root window object (for after method)
        """
        # If this GIF has no animation frames, don't start animation
        if class_label not in self.gif_frames or not self.gif_frames[class_label]:
            print(f"Debug: {class_label} has no animation frames, not starting animation")
            print(f"Debug: gif_frames keys = {list(self.gif_frames.keys())}")
            return
        
        # Check if animation is already running (same class_label and same Label)
        if (class_label in self.animation_jobs and 
            class_label in self.animation_labels and 
            self.animation_labels[class_label] == label_widget):
            # Animation already running, no need to restart
            print(f"Debug: Animation for {class_label} is already running, skipping restart")
            return
        
        print(f"Debug: Starting GIF animation for {class_label}, total {len(self.gif_frames[class_label])} frames")
        
        # Stop previous animation (if exists)
        self.stop_animation(class_label, root)
        
        # Store Label reference and root window
        self.animation_labels[class_label] = label_widget
        self.animation_indices[class_label] = 0
        
        # Start animation loop
        self._animate_gif(class_label, root)
    
    def stop_animation(self, class_label, root):
        """
        Stop GIF animation
        
        Args:
            class_label: Class label
            root: Tkinter root window object (for after_cancel)
        """
        if class_label in self.animation_jobs:
            after_id = self.animation_jobs[class_label]
            if after_id and root:
                try:
                    root.after_cancel(after_id)
                except:
                    pass
            del self.animation_jobs[class_label]
        
        if class_label in self.animation_labels:
            del self.animation_labels[class_label]
        
        if class_label in self.animation_indices:
            del self.animation_indices[class_label]
    
    def _animate_gif(self, class_label, root):
        """
        Internal method: Execute GIF animation loop
        
        Args:
            class_label: Class label
            root: Tkinter root window object
        """
        if class_label not in self.gif_frames or class_label not in self.animation_labels:
            print(f"Debug: Animation stopped - {class_label} not in gif_frames or animation_labels")
            return
        
        frames = self.gif_frames[class_label]
        delays = self.gif_delays.get(class_label, [100] * len(frames))
        label_widget = self.animation_labels[class_label]
        current_index = self.animation_indices.get(class_label, 0)
        
        if not frames or current_index >= len(frames):
            current_index = 0
        
        # Get current frame and convert to PhotoImage
        try:
            frame_image = frames[current_index]
            photo = ImageTk.PhotoImage(frame_image)
            
            # Update Label display
            label_widget.configure(image=photo)
            label_widget.image = photo  # Keep reference
            
            # Get current frame delay time (before updating index)
            delay = delays[current_index] if current_index < len(delays) else 100
            
            # Debug info (only first few times)
            if current_index < 3:
                print(f"Debug: Displaying frame {current_index}/{len(frames)}, delay {delay}ms")
            
            # Update index to next frame
            next_index = (current_index + 1) % len(frames)
            self.animation_indices[class_label] = next_index
            
            # Schedule next update (use wrapper function to avoid lambda closure issues)
            # Use copy of next_index to avoid closure issues
            next_index_copy = next_index
            def animate_callback():
                # Add debug info to confirm callback is called
                print(f"Debug: Callback executed - {class_label} next frame index {next_index_copy}")
                try:
                    self._animate_gif(class_label, root)
                except Exception as e:
                    print(f"Debug: Callback execution error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Set function name for tkinter recognition
            animate_callback.__name__ = f'_animate_gif_{class_label}'
            after_id = root.after(delay, animate_callback)
            self.animation_jobs[class_label] = after_id
            
            if next_index <= 3:  # Print debug info for first few frames
                print(f"Debug: Scheduled next frame {next_index}, delay {delay}ms, after_id = {after_id}")
            
        except Exception as e:
            print(f"Animated GIF update failed ({class_label}): {e}")
            import traceback
            traceback.print_exc()
            # If error, stop animation
            self.stop_animation(class_label, root)


def test_image_display():
    """Test image display functionality"""
    import tkinter as tk
    
    root = tk.Tk()
    root.title("Image Display Test")
    root.geometry("500x500")
    
    # Create image display manager
    image_display = ImageDisplay()
    
    # Test loading images
    label = tk.Label(root, text="Test Image Display")
    label.pack(pady=10)
    
    # Load Class 1 image
    photo1 = image_display.load_image('Class 1')
    if photo1:
        img_label1 = tk.Label(root, image=photo1)
        img_label1.image = photo1  # Keep reference
        img_label1.pack(pady=10)
    
    # Load Class 2 image
    photo2 = image_display.load_image('Class 2')
    if photo2:
        img_label2 = tk.Label(root, image=photo2)
        img_label2.image = photo2  # Keep reference
        img_label2.pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    test_image_display()
