import argparse
import cv2
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import logging

class EnhancedMedicalVideoProcessor:
    def __init__(self, video_path: str, frame_rate: int = 30):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the video file
            frame_rate: Number of frames to process per second
        """
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.cap = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the processor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def open_video(self) -> bool:
        """Open the video capture device."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video: {self.video_path}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error opening video: {str(e)}")
            return False
            
    def enhance_medical_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance medical image using CLAHE and Gaussian blur.
        
        Args:
            frame: Input frame to enhance
            
        Returns:
            Enhanced frame
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Remove noise using Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            return thresh
        except Exception as e:
            self.logger.error(f"Error in image enhancement: {str(e)}")
            return np.zeros_like(frame)
            
    def detect_text_regions(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect potential text regions in the frame using HSV color space.
        
        Args:
            frame: Input frame
            
        Returns:
            Mask of potential text regions
        """
        try:
            # Convert to HSV for better region detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for text-like regions
            lower = np.array([0, 0, 200])  # Bright regions
            upper = np.array([180, 50, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations to clean the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return mask
        except Exception as e:
            self.logger.error(f"Error in text region detection: {str(e)}")
            return np.zeros_like(frame)
            
    def process_frame(self, frame: np.ndarray) -> str:
        """
        Process a single frame to extract text.
        
        Args:
            frame: Input frame
            
        Returns:
            Extracted text
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return ""
            
    def process_video(self) -> List[str]:
        """
        Process the entire video and extract text from frames.
        
        Returns:
            List of extracted texts
        """
        if not self.open_video():
            self.logger.error("Failed to open video file")
            return []
            
        texts = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process every nth frame based on frame_rate
            if frame_count % (30 // self.frame_rate) == 0:
                text = self.process_frame(frame)
                if text.strip():
                    texts.append(text)
                    
            frame_count += 1
            
        self.cap.release()
        return texts

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Medical Video Text Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medical_video_processor.py -v input_video.mp4
  python medical_video_processor.py --video input_video.mp4 --frame-rate 15
"""
    )
    
    # Add arguments
    parser.add_argument(
        "-v", "--video",
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "-f", "--frame-rate",
        type=int,
        default=30,
        help="Number of frames to process per second"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create processor instance
    processor = EnhancedMedicalVideoProcessor(
        video_path=args.video,
        frame_rate=args.frame_rate
    )
    
    # Process video and get results
    results = processor.process_video()
    
    # Print results
    if results:
        print("\nExtracted text from video:")
        for i, text in enumerate(results, 1):
            print(f"\nFrame {i}:")
            print(text)
    else:
        print("No text found in the video.")

if __name__ == "__main__":
    main()
