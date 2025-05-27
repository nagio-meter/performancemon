import cv2
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class EnhancedMedicalVideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
    def enhance_medical_image(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Remove noise using Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def detect_text_regions(self, frame):
        # Convert to HSV for better region detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for text-like regions
        lower = np.array([0, 0, 200])  # Bright regions
        upper = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    def process_frame(self, frame):
        # Detect potential text regions
        region_mask = self.detect_text_regions(frame)
        
        # Enhance the frame
        enhanced_frame = self.enhance_medical_image(frame)
        
        # Apply mask to focus on text regions
        masked_frame = cv2.bitwise_and(enhanced_frame, enhanced_frame, mask=region_mask)
        
        # Extract text
        pil_image = Image.fromarray(masked_frame)
        text = pytesseract.image_to_string(pil_image, config='--psm 6')
        
        return text
    
    def process_video(self):
        texts = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process every nth frame
            if frame_count % 30 == 0:  # Adjust based on performance needs
                text = self.process_frame(frame)
                if text.strip():
                    texts.append(text)
                    
            frame_count += 1
                
        self.cap.release()
        return texts
