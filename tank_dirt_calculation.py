import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Dirt Area Estimation on Storage's Floating Roof")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        
        # Allow user to input the center and radius for cropping the circular region
        st.write("Define the circular region for accurate estimation:")
        center_x = st.slider("Center X", 0, image.width, image.width // 2)
        center_y = st.slider("Center Y", 0, image.height, image.height // 2)
        radius = st.slider("Radius", 0, min(image.width, image.height) // 2, min(image.width, image.height) // 4)
        
        # Convert image to numpy array and BGR format
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Crop the circular region
        circular_mask = np.zeros_like(image_np[:, :, 0])
        cv2.circle(circular_mask, (center_x, center_y), radius, 255, -1)
        
        circular_region = cv2.bitwise_and(image_np, image_np, mask=circular_mask)
        
        st.image(circular_region, caption='Cropped Circular Region', use_column_width=True)
        
        # Process the image
        dirt_percentage, result_image = process_image(circular_region, (center_x, center_y), radius)
        
        st.image(result_image, caption='Processed Image', use_column_width=True)
        st.write(f"Dirt Area Percentage: {dirt_percentage:.2f}%")

def process_image(image, center, radius):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Create a mask for the circular region
    circular_mask = np.zeros_like(gray)
    cv2.circle(circular_mask, center, radius, 255, -1)
    
    # Combine masks
    combined_mask = cv2.bitwise_and(thresh, circular_mask)
    
    # Calculate the total area of the circular region
    total_area = cv2.countNonZero(circular_mask)
    
    # Calculate the dirt area within the circular region
    dirt_area = cv2.countNonZero(combined_mask)
    
    # Calculate dirt area percentage
    dirt_percentage = (dirt_area / total_area) * 100
    
    # Create result image
    result_image = cv2.bitwise_and(image, image, mask=circular_mask)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return dirt_percentage, result_image

if __name__ == "__main__":
    main()
