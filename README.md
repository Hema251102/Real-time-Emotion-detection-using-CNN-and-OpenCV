# Real-time-Emotion-detection-using-CNN-and-OpenCV
 A real-time emotion detection system using CNN and OpenCV that captures facial expressions from live video input and classifies emotions like Happy, Sad, Angry, and Neutral using a trained deep learning model for facial expression recognition.
 I designed and implemented a real-time facial emotion recognition system that integrates computer vision with deep learning, capable of classifying human emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

The system is powered by a Convolutional Neural Network (CNN) trained on the FER-2013 dataset (48×48 grayscale facial images) and deployed using OpenCV for real-time video processing.

Architecture & Workflow
The workflow begins with live video capture via webcam, where OpenCV performs face detection using the HaarCascade classifier. Each detected face is cropped, resized to 48×48 pixels, and normalized to ensure optimal CNN performance.

The CNN follows a hierarchical feature extraction design:

First Conv2D Layer (32 filters) learns basic edges and contours.

Second Conv2D Layer (64 filters) detects facial parts such as eyes, mouth, and nose.

Third Conv2D Layer (128 filters) recognizes higher-level emotional patterns.
After flattening, a Dense layer with ReLU activation refines learned features, followed by a Softmax layer producing probability distributions across the seven emotions.

Training Details
Loss Function: Categorical Crossentropy (ideal for multi-class classification)

Optimizer: Adam (adaptive learning rate for faster convergence)

Regularization: Dropout(0.5) to reduce overfitting

Performance: Achieved ~80% training accuracy and ~78% validation accuracy, with stable convergence.

Deployment & Real-Time Performance
When deployed, the system processes frames at 20–25 FPS on a standard CPU. Each detected face is immediately classified, and the predicted emotion label is drawn on the video stream with a bounding box.

Applications
Such a system can be used in:

Human–Computer Interaction (adaptive interfaces based on user mood)

Mental Health Monitoring (emotion tracking over time)

Customer Experience Analysis (retail or service environments)

This project not only strengthened my understanding of CNNs, image preprocessing, and real-time computer vision but also enhanced my skills in end-to-end AI application development — from data preprocessing to model training and live deployment.
