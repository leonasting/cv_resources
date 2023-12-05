
 Object detection using a pre-trained model with OpenCV:

```
import cv2

def detect_objects(image, model):
    """
    Performs object detection on an image using a pre-trained model.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform object detection
    objects = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image
```


 Image segmentation using the GrabCut algorithm in OpenCV:
```
import cv2
import numpy as np

def segment_image(image, mask):
    """
    Segments an image using the GrabCut algorithm.
    The mask specifies the areas of the image to be considered as background (0), foreground (1), or unknown (2).
    """
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create a temporary mask to specify which areas are definitely background and which are definitely foreground
    temp_mask = np.zeros(image.shape[:2], np.uint8)
    temp_mask[mask == 0] = cv2.GC_BGD
    temp_mask[mask == 1] = cv2.GC_FGD
    
    # Perform GrabCut segmentation
    mask, bgd_model, fgd_model = cv2.grabCut(lab_image, temp_mask, None, None, 5, cv2.GC_INIT_WITH_MASK)
    
    # Refine the mask to get a binary mask
    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
    
    # Apply the mask to the original image
    segmented_image = image * mask[:, :, np.newaxis]
    
    return segmented_image
```