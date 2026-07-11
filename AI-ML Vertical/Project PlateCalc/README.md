Hi!

Operating Manual :

-> The actual working code is the ProjectPlateCalc and the Draft is the file that contains the files that were made in progress and contains errors in order to reach the final working code.

-> Run the Kaggle notebook on T4 GPU 

-> The code takes approx 2-3 hrs to run

-> Ensure that the Kaggle notebook does not deactivate during this duration.

About the Project : 

PlateCalc calculates the number of calories that the plate contains using just one photo of the meal. It uses a YOLOv8 instance segmentation model to segment each food object on the plate and then calculates it's portion weight and calorie content using the USDA FoodData Central API.

Tools Used :

-> Trained On : Kaggle's FoodSeg103, a 103-class food segmentation dataset

-> Model : YOLOv8 , trained with Ultralytics

-> Computer Vision : OpenCV, NumPy

-> Calorie Counter : USDA FoodData Central API

-> Executed on: Kaggle notebook, T4 GPUs

Workflow :

-> Find and examine the Kaggle FoodSeg103 dataset folder.

-> Convert semantic masks to YOLO polygons for each category and each image.

-> Create YOLO training and validation folders and create a data.yaml file.

-> Train the yolov8n-seg.pt model for 50 epochs with a 640px size and a batch of 16.

-> Run the model on new input to get the mask and class predictions.

-> Count item amounts based on components of discrete food items and calculate weight from the pixel area of amorphous food.

-> Get the nutritional information using the USDA API for each 100g of detected food.

-> Multiply the amount with calorie density to get the meal’s calories.
