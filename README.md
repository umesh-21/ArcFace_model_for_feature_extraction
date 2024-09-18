# Face Feature Extraction using ArcFace Model

This repository provides an implementation of face feature extraction using the **ArcFace** model for facial recognition tasks. The feature extraction is done through a function provided in the `feature_extraction.py` file, which uses the `ArcFace` model located inside the `networks` folder.


## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/face-feature-extraction.git
    cd face-feature-extraction
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

   Make sure to include a `requirements.txt` file containing necessary dependencies (e.g., PyTorch).

   
## Usage

You can use the `get_feature` function from `feature_extraction.py` to extract features from an input image.

### Example:

```python
from PIL import Image
import torch
from torchvision import transforms
from feature_extraction import get_feature

# Load and preprocess the image
image = Image.open('path_to_image.jpg')
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
input_image = transform(image).unsqueeze(0)  # Add batch dimension

# Extract features using ArcFace
features = get_feature(input_image)

# Print extracted feature shape
print("Extracted feature shape:", features.shape)
