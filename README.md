# DermiNOW: Dermatological Diagnosis and Treatment Platform

## Overview

DermiNOW is a comprehensive web application that combines modern machine learning techniques with traditional Ayurvedic principles to provide accurate diagnosis and personalized treatment plans for various skin conditions. It aims to empower individuals to take control of their skin health by bridging the gap between cutting-edge technology and ancient holistic healing.

## Features

- **Skin Condition Diagnosis**: Utilizes machine learning algorithms to analyze uploaded images and provide precise diagnoses for various skin conditions.
- **Ayurvedic Treatment Plans**: Offers personalized Ayurvedic treatment plans based on individual skin types, sensitivities, and medical history.
- **Doctor-Patient Communication**: Facilitates seamless communication between doctors and patients through secure messaging and consultation features.
- **Biopsy Image Analysis**: Allows doctors to upload and analyze biopsy images for further examination and diagnosis.
- **Prescription Management**: Enables doctors to prescribe medication and track patient prescriptions for effective treatment management.

## Installation

To run DermiNOW locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/DermiNOW.git
    cd DermiNOW
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the MongoDB database:**
    - Ensure MongoDB is installed and running on your machine.
    - Update the MongoDB connection URL in the configuration file as needed.

4. **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

## Usage

### For Patients

1. **Login**: Enter your credentials to log in as a patient.
2. **Image Upload**: Upload an image of your skin condition for diagnosis.
3. **Skin Health Report**: Fill out the skin health form to provide additional information for a comprehensive diagnosis.
4. **Ayurvedic Recommendations**: Answer questions to receive personalized Ayurvedic recommendations for your skin health.

### For Doctors

1. **Login**: Enter your credentials to log in as a doctor.
2. **Biopsy Analysis**: Upload biopsy images for detailed analysis and diagnosis.
3. **Prescription Management**: Input diagnosis and prescribe medication for patients.
4. **Patient Monitoring**: Monitor patient health and treatment progress through patient monitoring tools.

## Contributors

- **Rujuta Thombre** - [RujutaThombre](https://github.com/RujutaThombre)
- **Viveka Patil** - [Viveka9Patil](https://github.com/Viveka9Patil)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Libraries and Frameworks**: Streamlit, TensorFlow, Keras, OpenCV, Pandas.
- **Datasets**: Mock data generated for demonstration purposes.

## Contact

For any questions or feedback, please contact [rujuta.thombre@gmail.com](mailto:rujuta.thombre@gmail.com).
