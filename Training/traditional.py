import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class FaceSketchColorizer:
    def __init__(self):
        self.skin_tone = [172, 190, 215]
        self.eye_color = [30, 65, 120]
        self.lip_color = [120, 140, 200]
        self.blush_color = [150, 160, 225]
        self.hair_color = [30, 50, 80]
    
    def colorize(self, sketch_path, output_path=None):
        sketch = cv2.imread(sketch_path)
        if sketch is None:
            raise ValueError(f"Could not read image from {sketch_path}")
        
        if len(sketch.shape) == 3:
            gray_sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        else:
            gray_sketch = sketch
        
        height, width = gray_sketch.shape
        colorized = np.zeros((height, width, 3), dtype=np.uint8)
        
        _, sketch_binary = cv2.threshold(gray_sketch, 200, 255, cv2.THRESH_BINARY)
        sketch_binary_inv = cv2.bitwise_not(sketch_binary)
        
        kernel = np.ones((5, 5), np.uint8)
        face_mask = cv2.morphologyEx(sketch_binary_inv, cv2.MORPH_CLOSE, kernel, iterations=3)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_ERODE, kernel, iterations=1)
        
        colorized[face_mask > 0] = self.skin_tone
        
        eye_y = height // 3
        left_eye_x = width // 3
        right_eye_x = 2 * width // 3
        eye_radius = width // 15
        
        left_eye_mask = np.zeros((height, width), dtype=np.uint8)
        right_eye_mask = np.zeros((height, width), dtype=np.uint8)
        
        cv2.circle(left_eye_mask, (left_eye_x, eye_y), eye_radius, 255, -1)
        cv2.circle(right_eye_mask, (right_eye_x, eye_y), eye_radius, 255, -1)
        
        colorized[left_eye_mask > 0] = self.eye_color
        colorized[right_eye_mask > 0] = self.eye_color
        
        lips_y = int(height * 0.7)
        lips_width = width // 3
        lips_height = height // 10
        lips_x = width // 2 - lips_width // 2
        
        lips_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(lips_mask, (width // 2, lips_y), (lips_width, lips_height), 
                    0, 0, 180, 255, -1)
        
        for y in range(height):
            for x in range(width):
                if lips_mask[y, x] > 0:
                    colorized[y, x] = (0.7 * np.array(self.lip_color) + 0.3 * colorized[y, x]).astype(np.uint8)
        
        left_cheek_x = width // 4
        right_cheek_x = 3 * width // 4
        cheek_y = int(height * 0.45)
        cheek_radius = width // 8
        
        cheeks_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(cheeks_mask, (left_cheek_x, cheek_y), cheek_radius, 255, -1)
        cv2.circle(cheeks_mask, (right_cheek_x, cheek_y), cheek_radius, 255, -1)
        
        cheeks_mask = cv2.GaussianBlur(cheeks_mask, (21, 21), 0)
        
        for y in range(height):
            for x in range(width):
                if cheeks_mask[y, x] > 0:
                    intensity = cheeks_mask[y, x] / 255 * 0.3
                    colorized[y, x] = np.clip(
                        (1 - intensity) * colorized[y, x] + intensity * np.array(self.blush_color),
                        0, 255
                    ).astype(np.uint8)
        
        hair_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(hair_mask, (width // 2, height // 6), 
                    (width // 2, height // 3), 0, 0, 180, 255, -1)
        
        hair_only = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_mask))
        
        colorized[hair_only > 0] = self.hair_color
        
        sketch_lines = cv2.bitwise_not(sketch_binary)
        for c in range(3):
            colorized[:, :, c] = cv2.bitwise_and(colorized[:, :, c], cv2.bitwise_not(sketch_lines))
        
        colorized = cv2.GaussianBlur(colorized, (3, 3), 0)
        
        if output_path:
            cv2.imwrite(output_path, colorized)
        
        return colorized

    def display_comparison(self, sketch_path, colorized):
        sketch = cv2.imread(sketch_path)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title('Original Sketch')
        plt.imshow(sketch_rgb, cmap='gray')
        plt.axis('off')
        
        plt.subplot(122)
        plt.title('Colorized Face')
        plt.imshow(colorized_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    sketch_path = input("Enter the path to your face sketch image: ")
    output_dir = "outputs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(sketch_path)
    output_path = os.path.join(output_dir, f"colorized_{base_name}")
    
    try:
        colorizer = FaceSketchColorizer()
        colorized = colorizer.colorize(sketch_path, output_path)
        print(f"Colorized face image saved to {output_path}")
        
        colorizer.display_comparison(sketch_path, colorized)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()