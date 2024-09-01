import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('model.h5')

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Resize frame to the input size expected by your model
    resized_frame = cv2.resize(frame, (128, 128))  # Example size, adjust to your model's input size
    # Normalize the frame (scale pixel values to [0, 1] or other range as needed)
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match the input shape of the model (batch size, height, width, channels)
    processed_frame = np.expand_dims(normalized_frame, axis=0)
    return processed_frame

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    
    # Make predictions using the model
    predictions = model.predict(processed_frame)
    print(predictions)
    # Process and display predictions
    # For example, if your model outputs probabilities for classes:
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    if confidence >0.98 :
        # Display the prediction on the frame
        d = {0: "cat", 1: "dog"}
        cv2.putText(frame, f'Class: {d[predicted_class]}, Confidence: {confidence:.2f}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'i am not found any cat or dog', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Live Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
