# Real-Time Sign Language Recognition Project

## Overview

This project aims to bridge communication gaps by developing a Real-Time Sign Language Recognition system. Leveraging computer vision tools such as OpenCV, MediaPipe, and CVZone, alongside TensorFlow Keras for machine learning, the system interprets sign language gestures in real-time. Flask is employed to create a user-friendly interface, enhancing accessibility and ease of use.

## Project Structure

- **`data_collection`**: Contains scripts for capturing a diverse dataset of sign language gestures using OpenCV, MediaPipe, and CVZone.
  
- **`machine_learning`**: Utilizes TensorFlow Keras for training a robust model to recognize a wide array of sign language gestures.

- **`real_time_prediction`**: Implements real-time prediction capabilities using OpenCV, MediaPipe, CVZone, and Flask for a user-friendly interface.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Collection:**
   - Run the scripts in the `data_collection` directory to capture a diverse sign language dataset.

4. **Machine Learning:**
   - Train the machine learning model using the scripts in the `machine_learning` directory.

5. **Real-Time Prediction:**
   - Implement real-time prediction using the scripts in the `real_time_prediction` directory.

6. **Run Flask App:**
   - Navigate to `real_time_prediction` and run:
     ```bash
     python Realtime.py
     ```
   - Open your web browser and go to `http://localhost:5000` for the user-friendly interface.

## Future Improvements

1. Incorporate recognition for dual-hand gestures.
2. Enhance the user interface for better interaction.
3. Enrich the dataset with more images for improved accuracy in various conditions.
4. Embrace continuous iterative development based on user feedback.

## Contributions

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the [MIT License](LICENSE).