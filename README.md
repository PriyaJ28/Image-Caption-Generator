# Image-Caption-Generator
Generating Captions for images using Deep Learning

Output Samples:

![alt text](https://github.com/PriyaJ28/Image-Caption-Generator/blob/master/Untitled2.jpg?raw=true)
![alt text](https://github.com/PriyaJ28/Image-Caption-Generator/blob/master/Untitled3.jpg?raw=true)

Procedure:
1. Data Collection:
Data used in the project is Flickr 8k dataset which can be download by filling [this] https://forms.illinois.edu/sec/1713398 form provided by the University of Illinois at Urbana-Champaign.
This dataset contains 8000 images each with 5 captions.
These images are bifurcated as follows:
>Training Set — 6000 images
>Dev Set — 1000 images
>Test Set — 1000 images
Also training a model with large number of images may not be feasible on a system which is not a very high end PC/Laptop. For faster computation I used Google colab and ran the model in GPU.

2. Data understanding:
Read the file “Flickr8k.token.txt” which contains image_id each with 5 captions cleaned it for furthur use in the dict form where image_id is the key which maps to the list containing the 5 captions.

3. Data Cleaning

4. Vocabulary: 
Create a vocabulary of all the unique words present across all the 8000*5 (i.e. 40000) image captions (corpus) in the data set.
We write all these captions along with their image names in a new file namely, “descriptions.txt” and save it on the disk.

Made Model robust to outliers by ensuring that the words that occur more than 10 times in the captions are added to the vocabulary.

5.Load training and descriptions:
we add two tokens in every captions namely,
‘startseq’ -> This is a start sequence token which will be added at the start of every caption.
‘endseq’ -> This is an end sequence token which will be added at the end of every caption.

6. Data processing  - Images and Transfer Learning:
Images are imput in the model in the form of vectors. we need to convert every image into fixed size vector. For this purpose we opt TRANSFER LEARNING by using Xception model(CNN) created by Google Research.

This model was trained on Imagenet dataset to perform image classification on 1000 different classes of images. However, our purpose here is not to classify the image but just get fixed-length informative vector for each image. This process is called automatic feature engineering.

Hence, we just remove the last softmax layer from the model and extract a 2048 length vector (bottleneck features) for every image as follows:
![Automatic feature Engineering](https://miro.medium.com/max/2000/1*9VoYufkvd-hBxK3p2NEWmw.png)

We save all the bottleneck train features in a Python dictionary and save it on the disk using Pickle file, namely “encoded_train_images.pkl” whose keys are image names and values are corresponding 2048 length feature vector.Similarly we encode all the test images and save them in the file “encoded_test_images.pkl”.

7. Data Preprocessing:
During the training period, captions will be the target variables (Y) that the model is learning to predict. So,we will represent every unique word in the vocabulary by an integer (index). and we will also the find the maximum length of any caption.

8. Data Preparation:

Let us first see how the input and output of our model will look like. To make this task into a supervised learning task, we have to provide input and output to the model for training. We have to train our model on 6000 images and each image will contain 2048 length feature vector and caption is also represented as numbers. This amount of data for 6000 images is not possible to hold into memory so we will be using a generator method that will yield batches.

The generator will yield the input and output sequence.

For example:

The input to our model is [x1, x2] and the output will be y, where x1 is the 2048 feature vector of that image, x2 is the input text sequence and y is the output text sequence that the model has to predict.

![Partial Caption](https://miro.medium.com/max/1400/1*ME49hZnlJDtkA4cWtZjKNg.jpeg)

![Partial Caption wordtoix](https://miro.medium.com/max/1032/1*6G1eDpwq11eRY4rhD0yXPg.jpeg)

![Partial Caption after padding made each caption of max length](https://miro.medium.com/max/1032/1*gefPePe1I2-9pryw3axP1A.jpeg)

9. Model Architechture
![Model](https://miro.medium.com/max/2000/1*rfYN2EELhLvp2Van3Jo-Yw.jpeg)

![Detailed](https://miro.medium.com/max/2000/1*VGzSnYhyhpAAmGkSyOfeig.png)

Input_1 -> Partial Caption
Input_2 -> Image feature vector
Output -> An appropriate word, next in the sequence of partial caption provided in the input_1

10. Evaluation

