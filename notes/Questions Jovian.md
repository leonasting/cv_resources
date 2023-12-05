
## CV2


### 1. Write a function in Python that takes in an image and returns the number of red objects present in the image. You can assume that the image is in the RGB color space. How would you approach solving this problem?


```
import cv2
import numpy as np

def count_red_objects(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper boundaries for the red color range
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    upper_red2 = np.array([170, 50, 50])
    
    # Create a mask to identify pixels within the red color range
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_image, upper_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to remove noise from the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count the number of contours
    num_red_objects = len(contours)
    
    return num_red_objects
```

#### Pseudo Solution

In this code, the `count_red_objects` function takes in an image and performs the following steps:

1. Converts the image from the BGR color space to the HSV color space using `cv2.cvtColor()`.
2. Defines the lower and upper boundaries for the red color range in the HSV color space.
3. Creates a mask using `cv2.inRange()` to identify pixels within the red color range.
4. Applies morphological operations using `cv2.morphologyEx()` to remove noise from the mask.
5. Finds contours in the mask using `cv2.findContours()`.
6. Counts the number of contours found, which corresponds to the number of red objects in the image.
7. Returns the number of red objects.

#### Explanation
The HSV color space separates the color information (hue) from the intensity information (saturation and value), which makes it easier to isolate specific colors or color ranges. This is particularly useful when working with objects of a specific color, such as red in our case.

In the HSV color space, the hue component represents the color itself, while the saturation and value components represent the intensity and brightness, respectively. By converting the image to HSV, we can define a specific range of hue values that correspond to the desired color (in this case, red) and create a mask to isolate those pixels.

Compared to the RGB color space, which combines all three components (red, green, and blue) in a complex manner, working with the HSV color space simplifies the process of identifying and extracting specific colors from an image.

In our solution, by converting the image to the HSV color space, we can easily define the lower and upper boundaries for the red color range and create an appropriate mask to identify the red objects in the image.


2. Implement a convolutional neural network (CNN) in PyTorch for image classification. Assume that the input images are grayscale with dimensions 28x28 pixels and there are 10 classes to classify. Your network should consist of two convolutional layers followed by two fully connected layers. Use ReLU activation function for the hidden layers and softmax function for the output layer. Please write the code for this CNN model.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Apply first convolutional layer, followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.view(-1, 32 * 7 * 7)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax activation for classification
        x = F.softmax(x, dim=1)
        
        return x

# Create an instance of the CNN model
model = CNN()

# Print the model architecture
print(model)
```