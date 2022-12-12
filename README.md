# FACE MASK DETECTION

#### Group-Proposal
The Group-Proposal file was moved 12/12/2022 to Group-Proposal Folder but created 27 days ago from today.

#### Dataset
Link: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection 

## Code Folder
#### Pre-Trained
The pre_trained.py has the codes for training the dataset with Xception, ResNet50 and VGG16.
The input images and annotations path should have the correct path for data.

#### xception.py
This includes only the xception model as the best model from the pre-trained models in this project. It trains and test using xception.

#### Testing
The test_script.py test the pre-trained models and a part of ensemble models to combine predicting power but ensembling them was below
xception's accuracy.
PS: input_data_path, annotations_path must have the path to dataset's images and annotation's folder.

#### Custom CNN
The final_project.py has the codes for the custom made CNN network.

#### Testing for custom CNN (test_project.py)
This script is used to upload images from the internet and personal images to test the model.
