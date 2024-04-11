![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/f565d74a-91d1-411c-84aa-4c8e03ea6948)# Automatic Video Assistant Referee (VAR) for Soccer

## Overview

The project aims to implement an Automatic Video Assistant Referee (VAR) system in soccer matches to improve decision-making accuracy and fairness. It utilizes Convolutional Neural Networks (CNNs) for player, ball, and field detection, enabling automatic match management.

## How It Works

The system employs deep learning techniques for real-time analysis of soccer matches. It detects players, the ball, and key events such as offside and goal situations. The implementation integrates state-of-the-art algorithms like YOLO (You Only Look Once) for efficient object detection.

### Tools

- TensorFlow
- OpenCV
- YOLOv7
- Pytorch: Pytorch is a fully featured framework for building deep learning models, which is commonly used in applications like image recognition and language processing.
- Scikit-learn: Scikit-learn is a Python library used to implement various machine learning models for regression, classification, clustering, and statistical tools for analyzing these models.
- Numpy: Numpy is the fundamental package for scientific computing in Python, providing a multidimensional array object and various derived objects.
- Scikit-image: Scikit-image is a collection of algorithms for image processing, providing high-quality, peer-reviewed code written by an active community of volunteers.

## User Interface

We have implemented a website application that takes photos from users and the picture goes to the model, and the decision is made. There are two cases: Offside or not, and The ball passed goal line or not. We used HTML, CSS, and JavaScript to build the website. The main functions in the website are to take a photo from the user and provide feedback on the decision regarding offside or goal.

![Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/afa0cd95-700c-47da-9d46-898d9ce2265c))

## Offside Case

The code provided seems to be a part of an object detection system that detects objects in an image and assigns them team labels based on the color of their jerseys. There are several functions involved in this system:

1. "remove background": Removes the background from the image using morphological operations and contour detection.
![Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/99e9182c-2a70-445f-90f0-97ca22eeafab))
2. "run inference": Resizes and pads the resulting image to a fixed size and applies further transformations.
![Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/99e9182c-2a70-445f-90f0-97ca22eeafab))
3. "extract jersey colors": Extracts features related to the object's jersey color.
![Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/c86cf8b0-905b-4db5-a497-5860d2b010c7))
4. "assign jersey labels": Assigns team labels (0 or 1) to each object based on their jersey colors.
5. "plot results": Plots the original image and the predicted image with labels.
![Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/8780ad7e-4f64-4cf8-a256-e058fa1c51c2))
The attacking team is determined by comparing the farthest positions of the players in each team.

### Example Image: Offside Case
![Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/9c56e16d-6778-444a-a2c5-dc24f2630116))

## Goal Detection Case

The goal detection model is a deep learning model that uses a convolutional neural network (CNN) to classify images as either a goal or not a goal. The model is trained on a dataset of images that show a ball either inside or outside the goalpost and learns to distinguish between the two classes based on patterns in the images.

The model architecture consists of several convolutional layers, followed by pooling layers and fully connected layers. During training, the model is shown a set of labeled images and adjusts its weights to minimize the difference between its predictions and the true labels.

Once the model is trained, it can be used to classify new images as either a goal or not a goal by feeding the image into the model and getting a probability score between 0 and 1. The goal detection model has various potential applications, such as in sports analytics to automatically detect goals in soccer matches.

### Example Image: Goal Detection Case
![Goal Detection Ofsside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/a87a2403-4904-4a35-823f-2cffd2c19a10))
![Goal Detection Not Offside Example](![image](https://github.com/Avatar2001/Automated-Assistant-Video-Referee-VAR-/assets/71982844/8ca1451d-f7b1-42aa-b89f-220dfba8a4ab))

## Conclusion

The implementation of the automatic video assistant referee will bring significant changes to the football sport, particularly in terms of how referees make decisions during matches. With the help of technology, referees are now able to review incidents such as fouls, offside decisions, and goals, to ensure that the correct decision is made.

Moreover, automatic VAR has also played a crucial role in improving the transparency and fairness of football matches. The system allows for greater accountability and accuracy, reducing the chances of a team being disadvantaged by a poor decision. This has not only enhanced the overall quality of the game but also increased the level of trust between players, fans, and referees.

While there has been some criticism of the system, particularly around the time taken to review decisions, it is important to note that VAR is still a relatively new technology in football. As the system continues to evolve and become more efficient, we can expect to see a reduction in the time taken to review decisions, and an improved overall experience for players, fans, and officials alike.

In conclusion, the video assistant referee is a significant technological advancement in football, one that has brought about many positive changes to the game. As the technology continues to evolve, we can expect to see even more improvements in the future, further enhancing the transparency, accuracy, and fairness of football matches.

## References

- [UEFA's refereeing organization]([#](https://www.marca.com/en/football/international-football/2018/03/03/5a9ac695268e3e265d8b45af.html))
- [New deep learning techniques analyze athletes' decision-making]([#](https://www.sciencedaily.com/releases/2017/03/170306092708.htm))
- [Machine Learning in Modeling High School Sport Concussion Symptom Resolve]([#](https://journals.lww.com/acsm-msse/Fulltext/2019/07000/Machine_Learning_in_Modeling_High_School_Sport.2.aspx))
- [An MLP-based player detection and tracking in broadcast soccer video](#)
- [Video-Based Soccer Ball Detection in Difficult Situations](#)
- [DeepPlayer-Track: Player and Referee Tracking with Jersey Color Recognition in Soccer](#)
- [FootAndBall: Integrated Player and Ball Detector](#)
- [Semi-Supervised Training to Improve Player and Ball Detection in Soccer](#)
- [DETECTING OFFSIDE POSITION USING 3D RECONSTRUCTION](#)
- [Automated Offside Detection by Spatio-Temporal Analysis of Football Videos](#)
- [Virtual lines for offside situations analysis in football](#)
- [Football and Computer Vision](#)
- [Deep Learning-Based Football Player Detection in Videos](#)
- [Dive Into Football Analytics with TensorFlow Object Detection API](#)
- [YOLOv7 explanation and implementation from scratch](#)
