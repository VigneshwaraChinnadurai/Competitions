Link:  https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-emotion-detection-tom-jerry-cartoon/machine-learning/detect-emotions-of-your-favorite-toons-7d2c0f23/


(Problem: Multi class classification with Imbalanced dataset)

Hi All,

As the input is given as video, we need to first break them down into images with FPS using Video2Image.py file. 

As I am using ImageDataGenerator method, seperating the images according to the classes using Train_classification.py file.

Let's jump into the main part. (Final_model.py) (Note: Instruction given in step by step)

Importing necessary libraries used.

4 sets of 2 convolutional layers, a maxpolling layer and a dropout layer is stacked to form the network and at last 2 have batch normalization layer in addition to normalize the outputs.

Followed by a flattern layer and 2 dense layer and a dropout layer which completes the architecture.

Combining relu and softmax activation functions which provided best results.

Compiling the layers with Accuracy as major metrics by setting categorical cross entropy as loss function and adam optimizer

With the Image data generator, making the small dataset bigger using the arguments such as shear range, zoom range, horizontal flip and etc.

With the flow from directory method, we feeded the model with inputs.

Used early stop and model check point for my convenience and higher performance in a short span of time.

As the data is imbalanced data, setting weight accordingly and fitting the model with train data.

with the best model generated using Model checkpoint, loaded that model and predicted the output.

Thanks for reading this.

Have a good day.
