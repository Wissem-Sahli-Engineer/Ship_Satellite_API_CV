from pathlib import Path
# pyrefly: ignore [missing-import]
from tf_keras.models import load_model  # Teachable Machine H5 models require tf_keras
import cv2  # Install opencv-python
# pyrefly: ignore [missing-import]
import numpy as np
from utils import format

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("models/model_2/keras_model.h5", compile=False)

# Load the labels
class_names = open("models/model_2/labels.txt", "r").readlines()

ship_path = Path("data/no_ship")

errors = 0

for file in ship_path.iterdir():
    
    img = cv2.imread(file)

    # Input Format
    image = format(img)

    """
    # Show the image in a window
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    if index == 0:
        errors += 1

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

print(errors)
print(f'Accuracy : {1 - errors/1000}')
