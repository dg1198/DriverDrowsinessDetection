# Driver Drowsiness Detection

## Overview

Driver Drowsiness Detection is a real-time application designed to monitor driver alertness and detect signs of drowsiness. Using computer vision techniques, the system analyzes the driver’s eyes and triggers an alarm if drowsiness is detected, enhancing road safety.

## Features

- Real-time detection of driver drowsiness based on eye aspect ratio (EAR)
- Audio alert to notify the driver when drowsiness is detected
- Visual display of the EAR and drowsiness status

## Technologies Used

- Python
- OpenCV
- NumPy
- SciPy
- Pygame

## Requirements

To run this project, you will need the following Python packages:

- `opencv-python`
- `numpy`
- `scipy`
- `pygame`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dg1198/DriverDrowsinessDetection.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd DriverDrowsinessDetection
   ```

3. Install the required dependencies:
   ```bash
   pip install opencv-python numpy scipy pygame
   ```

4. Ensure the following files are present in the project directory:
   - `alarm.wav`: The sound file played during a drowsiness alert.
   - `haarcascade_lefteye_2splits.xml`: Haar cascade for left eye detection.
   - `haarcascade_righteye_2splits.xml`: Haar cascade for right eye detection.
   - `haarcascade_frontalface_default.xml`: Haar cascade for face detection (included with OpenCV).

## Usage

To run the application, execute the following command:

```bash
python detect.py
```

Make sure your webcam is connected and accessible. The program will open a window displaying the video feed. 

- Press the `q` key to exit the application.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
