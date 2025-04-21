# SVM Classifiers for Hand-Based Subject Identification
This repository contains the implementation of an SVM classifier system designed to identify the identity of various subjects based on hand images. By leveraging various SVM classifiers and different feature extraction techniques such as LBP and HOG, the system focuses on the morphology, texture of the hand, and palm lines.
The project can handle identification in the following ways:
- open set with and without impostor
- closed set
The project was developed as part of the "Biometric Systems" course during the Master's degree in Computer Science at Sapienza University of Rome in the 2024/25 academic year.

## 🖥️ Steps

1. Clone the repository

2. Install dependencies

3. Download the 11k Hands dataset
   - [Dataset Link](https://sites.google.com/view/11khands)

4. In the main file, set the necessary values for execution:
     - num_exp: Number of epochs
     - image_path: Path to the folder containing the images of the 11kHands dataset or the pre-cutted images
     - cav_path: Path to the file containing the metadata of the 11kHands dataset images
     - num_sub: Numbers of subjects to be extracted for identification
     - num_img: Total number of images to be extracted for each subject
     - isClosedSet: Identifies whether identification should occur in OpenSet or ClosedSet mode
     - num_impostor: Possible number of impostors to be included in the identification, included in the total number of subjects

5. Run the main script


## 🧾📈 More details and Performance evaluation
For more details on the implementation and its performance, you can refer to the paper [HandsIdentityRecognition](https://github.com/patriziorenelli/HandsBasedRecognition/blob/main/HandsIdentityRecognition.pdf)

# 🧑‍💻 Collaborators

- Matteo Rocco
- Alessio Taruffi
- Nicolò Candita
