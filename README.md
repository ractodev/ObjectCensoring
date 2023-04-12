# ObjectCensoring
This application allows the user to censor objects in a image using a trained Artificial Neural Network. This project has been a part of the course DD2380 Artificial Intelligence: https://www.kth.se/student/kurser/kurs/DD2380?l=en.

## How to run:
Prerequisites:
* TensorFlow 1.x (I have used TensorFlow 1.15 when testing and it worked great. Version 2 is rumored to not work with the application, so use 1.x to be on the safe side.)
* Python (TensorFlow 1.x has worked without consequences for me on Python 3.6.9. I am unsure if the program will work with newer versions than Python 3.6.x)
* Some packages needs to be installed using PIP (the console will prompt you on the needed packages that needs to be installed).

### Step 1.
Add the image(s) you want to use into the project-folder. Be sure to convert the pictures to .jpg-format.

### Step 2.
Compile and run `blur_object.py`. The program will prompt you to choose an input. Enter the name of the image in the console excluding the extension (.jpg).
*For example: If you want to use a picture called `person_and_dog.jpg`, the input argument should be specified as: "person_and_dog".*

### Step 3.
The program will show you the segmentation map and what objects it has detected to the far right in the picture. 

### Step 4.
Once the segmentation map has been closed down, the program will prompt the user to choose what object type that should be censored.
Select the one of the objects that showed up on the segmentation map. Otherwise: the console will print out all objects types that 
the user can choose from. However, if the selected object type is not detected by the segmentation process, nothing in the image will
be censored.
*For example: The program detects a person-object and a dog-object in the image the user selected as input. In this case, you can enter "person" or "dog" in the console to censor the objects.*

### Final step.
The resulting image with censored objects will be located in the project-folder. This image will be named `result.jpg`.
