import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys
from tensorflow.lite.python.interpreter import Interpreter

# Add to_do directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'to_do'))
try:
    from image_display import ImageDisplay
except ImportError:
    print("Warning: Unable to import image_display module")
    ImageDisplay = None

class WebcamClassifier:
    def __init__(self):
        # Load labels file
        self.labels = self.load_labels("model/labels.txt")
        
        try:
            # Load TFLite model
            model_path = "model/model_unquant.tflite"
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"Model loading failed: {e}")
            # Create a simple error dialog
            error_root = tk.Tk()
            error_root.title("Error")
            error_root.geometry("400x200")
            ttk.Label(error_root, text=f"Model loading failed:\n{e}", font=("Arial", 12)).pack(pady=20)
            ttk.Button(error_root, text="OK", command=error_root.destroy).pack()
            error_root.mainloop()
            raise e
        
        # Get model input/output information
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        
        # Create GUI window
        self.root = tk.Tk()
        self.root.title("Webcam Classifier")
        self.root.geometry("800x600")
        
        # Set window properties
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.resizable(True, True)
        
        # Create controls
        self.setup_gui()
        
        # Control variables
        self.is_running = False
        self.capture_thread = None
        self.cap = None  # Camera object
        self.current_camera_index = 0  # Current camera index
        self.available_cameras = []  # Available cameras list
        self.is_high_confidence = False  # Whether accuracy exceeds 80%
        self.current_prediction = None  # Current prediction result
        self.last_displayed_label = None  # Last displayed label, to avoid reloading
        self.low_confidence_count = 0  # Debounce: only clear reference image after N consecutive low-confidence updates
        
        # GIF tick (same pattern as test_gif_standalone.py)
        self._gif_after_id = None
        self._gif_index = [0]
        self._gif_photo_frames = None
        self._gif_delays = None
        self._gif_ref_label = None
        
        # Image display manager
        if ImageDisplay:
            self.image_display = ImageDisplay(base_path=os.path.join(os.path.dirname(__file__), 'to_do', 'image'))
        else:
            self.image_display = None
        
    def load_labels(self, labels_path):
        """Load labels from labels.txt file"""
        labels = []
        try:
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Handle format: "0 Class 1" or "1 Class 2"
                            parts = line.split(' ', 1)
                            if len(parts) >= 2:
                                label_name = parts[1]
                                labels.append(label_name)
                            else:
                                labels.append(line)
                print(f"Successfully loaded {len(labels)} labels: {labels}")
            else:
                print(f"Labels file does not exist: {labels_path}")
                labels = ['Unknown Class']
        except Exception as e:
            print(f"Failed to load labels file: {e}")
            labels = ['Unknown Class']
        
        return labels if labels else ['Unknown Class']
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.start_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_capture)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_capture, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Camera selection
        ttk.Label(control_frame, text="Camera:").grid(row=0, column=2, padx=(20, 5))
        self.camera_var = tk.StringVar(value="Camera 0")
        self.camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, 
                                         state="readonly", width=15)
        self.camera_combo.grid(row=0, column=3, padx=(0, 10))
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_selected)
        
        self.switch_camera_btn = ttk.Button(control_frame, text="Switch Camera", command=self.switch_camera)
        self.switch_camera_btn.grid(row=0, column=4, padx=(0, 10))
        
        # Detect available cameras
        self.detect_available_cameras()
        
        # Model information display
        model_info_frame = ttk.Frame(main_frame)
        model_info_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.model_info_label = ttk.Label(model_info_frame, text=f"Model: model_unquant.tflite | Labels: {len(self.labels)}")
        self.model_info_label.grid(row=0, column=0, padx=(0, 10))
        
        # Camera and image display area (side by side)
        camera_image_frame = ttk.Frame(main_frame)
        camera_image_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera display area
        camera_frame = ttk.LabelFrame(camera_image_frame, text="Camera Feed", padding="5")
        camera_frame.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = tk.Label(camera_frame, text="Waiting to start camera...", bg="black", fg="white")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Reference image opens in separate window; placeholder here
        ttk.Label(camera_image_frame, text="Reference image opens in separate window\nwhen accuracy >= 80%", 
                 font=("", 10)).grid(row=0, column=1, padx=20, pady=20)
        
        # Reference image window (Toplevel) - created on first use
        self.reference_window = None
        self.reference_image_label = None
        self.reference_image_photo = None
        
        # Configure column weights
        camera_image_frame.columnconfigure(0, weight=1)
        camera_image_frame.columnconfigure(1, weight=0)
        camera_image_frame.rowconfigure(0, weight=1)
        
        # Classification results display
        result_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.result_text = tk.Text(result_frame, height=8, width=60)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Image display area expandable
        main_frame.rowconfigure(3, weight=1)  # Classification results area expandable
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
    def on_camera_selected(self, event=None):
        """Called when camera selection changes"""
        selected_text = self.camera_var.get()
        if selected_text and selected_text != "No camera found":
            try:
                # Extract index from text (e.g., "Camera 0" -> 0)
                index = int(selected_text.split()[-1])
                if index in self.available_cameras:
                    self.current_camera_index = index
                    print(f"Selected camera index: {index}")
                    self.status_var.set(f"Selected camera index: {index}")
            except (ValueError, IndexError):
                pass
    
    def switch_camera(self):
        """Switch camera"""
        if self.is_running:
            # If capturing, stop first
            was_running = True
            self.stop_capture()
        else:
            was_running = False
        
        # Re-detect available cameras
        self.detect_available_cameras()
        
        # If was running before, restart
        if was_running and self.available_cameras and self.available_cameras[0] != -1:
            self.start_capture()
        else:
            self.status_var.set("Camera list refreshed")
    
    def _get_reference_window(self):
        """Create reference image Toplevel on first use; return (window, label)."""
        if self.reference_window is not None and self.reference_window.winfo_exists():
            return self.reference_window, self.reference_image_label
        self.reference_window = tk.Toplevel(self.root)
        self.reference_window.title("Reference Image")
        self.reference_window.geometry("320x320")
        self.reference_window.resizable(True, True)
        self.reference_window.protocol("WM_DELETE_WINDOW", self._hide_reference_window)
        frame = tk.Frame(self.reference_window, padx=10, pady=10, bg="white")
        frame.pack(expand=True, fill=tk.BOTH)
        self.reference_image_label = tk.Label(frame, text="Waiting for detection...", bg="lightgray", anchor="center")
        self.reference_image_label.pack(expand=True, fill=tk.BOTH)
        self.reference_image_photo = None
        self.reference_window.withdraw()  # Hide until we have an image to show
        return self.reference_window, self.reference_image_label

    def _stop_gif_tick(self):
        """Stop GIF tick loop (same pattern as test_gif_standalone.py)."""
        if self._gif_after_id is not None:
            try:
                self.root.after_cancel(self._gif_after_id)
            except Exception:
                pass
            self._gif_after_id = None
        self._gif_ref_label = None

    def _start_gif_tick(self, photo_frames, delays, ref_label):
        """Start GIF tick loop like test_gif_standalone.py."""
        self._stop_gif_tick()
        if not photo_frames or not ref_label:
            return
        self._gif_photo_frames = photo_frames
        self._gif_delays = delays
        self._gif_ref_label = ref_label
        self._gif_index[0] = 0

        def tick():
            if self._gif_ref_label is None or not self._gif_ref_label.winfo_exists():
                self._gif_after_id = None
                return
            photo_frames = self._gif_photo_frames
            delays = self._gif_delays or [100] * len(photo_frames)
            i = self._gif_index[0] % len(photo_frames)
            self._gif_ref_label.configure(image=photo_frames[i])
            self._gif_ref_label.image = photo_frames[i]
            self._gif_index[0] += 1
            delay = delays[i] if i < len(delays) else 100
            self._gif_after_id = self.root.after(delay, tick)

        self._gif_after_id = self.root.after(100, tick)

    def _hide_reference_window(self):
        """Hide reference window (user closed it); stop animations."""
        self._stop_gif_tick()
        if self.reference_window is not None and self.reference_window.winfo_exists():
            self.reference_window.withdraw()

    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            # If capturing, stop first
            self.stop_capture()
        
        # Confirm if really want to close
        if tk.messagebox.askokcancel("Exit", "Are you sure you want to exit the application?"):
            if self.reference_window is not None and self.reference_window.winfo_exists():
                self.reference_window.destroy()
            self.root.destroy()
    
    def detect_available_cameras(self):
        """Detect all available cameras"""
        print("Detecting available cameras...")
        self.available_cameras = []
        camera_names = []
        
        for index in range(10):  # Check indices 0-9
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Wait for camera initialization
                time.sleep(0.2)
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.available_cameras.append(index)
                    camera_names.append(f"Camera {index}")
                    print(f"Found available camera, index: {index}")
                cap.release()
        
        if not self.available_cameras:
            camera_names = ["No camera found"]
            self.available_cameras = [-1]
        
        # Update dropdown
        self.camera_combo['values'] = camera_names
        if self.available_cameras:
            self.current_camera_index = self.available_cameras[0]
            self.camera_var.set(camera_names[0])
        
        print(f"Detected {len(self.available_cameras)} available cameras: {self.available_cameras}")
    
    def find_available_camera(self, camera_index=None):
        """Find available camera (using specified index or current index)"""
        if camera_index is None:
            camera_index = self.current_camera_index
        
        print(f"Opening camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Wait for camera initialization
            time.sleep(0.5)
            # Try to read a few frames
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Camera index {camera_index} is available")
                    return cap, camera_index
                time.sleep(0.1)
            cap.release()
        return None, -1
    
    def start_capture(self):
        """Start camera capture"""
        print("Starting camera capture...")
        try:
            # Update UI state first
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set("Searching for camera...")
            self.root.update()  # Force GUI update
            
            # Find available camera (using currently selected index)
            self.cap, camera_index = self.find_available_camera(self.current_camera_index)
            
            if self.cap is None:
                raise Exception(f"Unable to open camera index {self.current_camera_index}. Please check:\n1. Is the camera connected?\n2. Is the camera being used by another program?\n3. Are the camera drivers working properly?")
            
            self.current_camera_index = camera_index
            print(f"Camera opened, index: {camera_index}")
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Set buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test reading again
            self.status_var.set("Testing camera...")
            self.root.update()
            
            retry_count = 0
            test_success = False
            while retry_count < 10 and not test_success:
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    test_success = True
                    print(f"Camera test successful (attempt {retry_count + 1})")
                else:
                    retry_count += 1
                    time.sleep(0.2)
            
            if not test_success:
                raise Exception("Unable to read from camera. Please check if the camera is working properly or being used by another program")
            
            print("Camera test successful, starting capture loop...")
            
            self.status_var.set("Capturing camera feed...")
            
            # Clear previous display content
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Capturing, please wait...")
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            print("Capture thread started")
            
        except Exception as e:
            error_msg = f"Failed to start camera: {e}"
            print(f"Error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            self.is_running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped")
        
        # Release camera resources
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear image display area
        self.clear_image_display()
    
    def capture_loop(self):
        """Camera capture loop"""
        print("Capture loop started")
        error_count = 0
        frame_count = 0
        
        while self.is_running:
            try:
                if self.cap is None:
                    raise Exception("Camera object is None")
                
                if not self.cap.isOpened():
                    raise Exception("Camera is not opened")
                
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Try to reopen camera (using currently selected index)
                    print(f"Read failed, trying to reopen camera index {self.current_camera_index}...")
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(self.current_camera_index)
                    if not self.cap.isOpened():
                        raise Exception("Camera connection lost")
                    time.sleep(0.5)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        raise Exception("Unable to read camera feed")
                
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Processed {frame_count} frames")
                
                # Preprocess image
                processed_img = self.preprocess_image(frame)
                
                # Perform inference
                prediction = self.classify_image(processed_img)
                
                # Update GUI (in main thread)
                # Ensure frame is not None
                if frame is not None and frame.size > 0:
                    self.root.after(0, self.update_gui, frame.copy(), prediction)
                else:
                    print("Warning: Frame is empty, skipping GUI update")
                
                # Reset error count
                error_count = 0
                
                # Control frame rate
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                error_count += 1
                print(f"Capture error #{error_count}: {e}")
                import traceback
                traceback.print_exc()
                
                # If consecutive errors exceed 5, stop capture
                if error_count >= 5:
                    print("Too many consecutive errors, stopping capture...")
                    self.root.after(0, lambda: self.status_var.set(f"Too many capture errors, stopped (error #{error_count})"))
                    self.root.after(0, self.stop_capture)
                    break
                else:
                    time.sleep(1)
        
        print("Capture loop ended")
    
    def preprocess_image(self, image):
        """Preprocess image to match model input"""
        # Resize image to match model input
        target_size = (self.input_shape[1], self.input_shape[2])
        resized = cv2.resize(image, target_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def classify_image(self, image):
        """Classify image"""
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], image)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get top-3 predictions
            top_indices = np.argsort(output_data[0])[-3:][::-1]
            predictions = []
            
            for idx in top_indices:
                confidence = float(output_data[0][idx])
                label = self.labels[idx] if idx < len(self.labels) else f"Class {idx}"
                predictions.append((label, confidence))
            
            # Check if highest confidence exceeds 80%
            if predictions and len(predictions) > 0:
                top_confidence = predictions[0][1]  # Highest confidence
                self.is_high_confidence = top_confidence >= 0.80
                self.current_prediction = predictions[0]  # Store highest confidence prediction
            else:
                self.is_high_confidence = False
                self.current_prediction = None
            
            return predictions
            
        except Exception as e:
            print(f"Inference error: {e}")
            self.is_high_confidence = False
            self.current_prediction = None
            return [("Error", 0.0)]
    
    def update_gui(self, image, prediction):
        """Update GUI display"""
        try:
            if image is None or image.size == 0:
                print("Warning: Received empty image")
                return
            
            # Update image display
            # Resize image to fit display (maintain aspect ratio)
            height, width = image.shape[:2]
            max_width, max_height = 640, 480
            
            # Calculate scale factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            display_img = cv2.resize(image, (new_width, new_height))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL image
            pil_img = Image.fromarray(display_img)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update label - ensure image display
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep reference to prevent garbage collection
            
            # Force GUI update
            self.image_label.update_idletasks()
            
            # Update classification results
            self.result_text.delete(1.0, tk.END)
            timestamp = time.strftime("%H:%M:%S")
            result_str = f"[{timestamp}] Classification Results:\n\n"
            
            for i, (label, confidence) in enumerate(prediction):
                result_str += f"{i+1}. {label}: {confidence:.2%}\n"
            
            # Display high accuracy status
            result_str += f"\nAccuracy Status: "
            if self.is_high_confidence:
                result_str += f"✓ High Accuracy (≥80%)\n"
                result_str += f"Current Prediction: {self.current_prediction[0]} ({self.current_prediction[1]:.2%})"
            else:
                result_str += f"✗ Low Accuracy (<80%)"
            
            self.result_text.insert(tk.END, result_str)
            
            # Update reference image display
            self.update_reference_image()
            
            # Update status bar
            status_text = f"Capturing - Last Update: {timestamp}"
            if self.is_high_confidence:
                status_text += f" | High Accuracy: {self.current_prediction[0]} ({self.current_prediction[1]:.2%})"
            self.status_var.set(status_text)
            
        except Exception as e:
            print(f"GUI update error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_reference_image(self):
        """Update reference image display in separate window based on classification results"""
        try:
            if not self.image_display:
                return
            
            # If accuracy exceeds 80%, display corresponding class image in reference window
            if self.is_high_confidence and self.current_prediction:
                self.low_confidence_count = 0
                class_label = self.current_prediction[0]
                
                if class_label != self.last_displayed_label:
                    self._stop_gif_tick()
                    photo = self.image_display.load_image(class_label, max_width=320, max_height=320)
                    if photo:
                        win, ref_label = self._get_reference_window()
                        win.deiconify()
                        ref_label.configure(image=photo, text="")
                        self.reference_image_photo = photo
                        if class_label in self.image_display.gif_frames:
                            photo_frames = self.image_display.gif_frames[class_label]
                            delays = self.image_display.gif_delays.get(class_label, [100] * len(photo_frames))
                            self._start_gif_tick(photo_frames, delays, ref_label)
                        self.last_displayed_label = class_label
                    else:
                        win, ref_label = self._get_reference_window()
                        ref_label.configure(image="", text=f"Image load failed\n{class_label}")
                        self.reference_image_photo = None
                        self.last_displayed_label = None
            else:
                if self.last_displayed_label is not None:
                    self.low_confidence_count += 1
                    if self.low_confidence_count >= 5:
                        self._stop_gif_tick()
                        self.last_displayed_label = None
                        if self.reference_window is not None and self.reference_window.winfo_exists():
                            self.reference_window.withdraw()
                        self.reference_image_photo = None
                
        except Exception as e:
            print(f"Update reference image error: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_image_display(self):
        """Clear image display area"""
        try:
            self.image_label.configure(image="", text="Waiting to start camera...")
            self.image_label.image = None
            
            self._hide_reference_window()
            self.reference_image_photo = None
            self.last_displayed_label = None
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Waiting to start camera...")
            
        except Exception as e:
            print(f"Clear display error: {e}")
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            # Ensure camera resources are released
            if self.cap:
                self.cap.release()

def main():
    try:
        app = WebcamClassifier()
        app.run()
    except Exception as e:
        print(f"Application startup error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
