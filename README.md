                            IMAGE FORGERY DETECTION
                            
Welcome to the Image Forgery Detection project! This system allows users to upload an image via a web interface, and it analyzes the image to detect any signs of tampering using advanced deep learning techniques. The results are displayed back to the user on the same webpage.

 ##Features
->Image Upload: Users can upload an image directly through a user-friendly web page.
->Forgery Detection: The system analyzes the image at the pixel level to determine whether it has been tampered with.
->Results Display: The detection results are displayed back to the webpage, indicating whether the image is genuine or tampered.

##Dataset
The model was developed using the CASIA-V2 dataset, which contains a total of 12,246 images:
Genuine Images: 7,123
Tampered Images: 5,123
-> The Dataset can be downloaded from Kaggle by tapping on the below link:
https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset

##Technology Stack
#Front End:
--> HTML, CSS, Flask
#Back End:
-->Python, Deep Learning, Convolutional Neural Networks (CNN),Error Level Analysis (ELA)
#Deep Learning Frameworks:
1) TensorFlow
2) Keras
   
#Libraries:
->PIL (Python Imaging Library), Scikit-Image (skimage), OpenCV (cv2)

              INSTALLATION
#Prerequisites
->Python 3.8 or higher, Flask,TensorFlow, Keras, PIL, Scikit-Image, OpenCV

#STEPS
1) Clone the Repository:
git clone https://github.com/your-username/image-forgery-detection.git

2) Install Dependencies:
Navigate to the project directory and install the required dependencies using pip:
cd image-forgery-detection
pip install -r requirements.txt

3)Download Dataset:
Download the CASIA-V2 dataset from the official source.
Place the dataset in the data directory within the project.

4) Develop the Deep Leraning Model
-> After installing the required dependencies
-> Develop the deep learning model using the DataSet
-> Connect the Model to the frontend
   
5) Run the Flask Application:
-> python main.py
   
6) Access the Web Interface:
Open your web browser and navigate to http://localhost:5000.


###USAGE
1)Upload an Image:
Visit the webpage and use the upload button to select an image from your device.

2)Run Detection:
Click on the "Detect Forgery" button to analyze the image.

3)View Results:
The system will process the image and display the result on the same page, indicating whether the image is genuine or tampered.


                                 Acknowledgements
Dataset: CASIA-V2, developed by the Institute of Automation, Chinese Academy of Sciences.
Libraries: Thank you to the developers of TensorFlow, Keras, PIL, Scikit-Image, and OpenCV.
