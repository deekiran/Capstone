# Capstone
CNN Model for detecting Glaucoma and Diabetic Retinopathy 
Data set of DR from Kaggle : https://www.kaggle.com/c/diabetic-retinopathy-detection

Steps for Image processing </br>
1 Remove dark and corrupted Images </br>
2 Resize and crop Images to extract Region Of Interest.</br>
3 As you can see in the below image, the count of each class has wide range. Thus,  DR diagnosed Images are rotated at 60,90,120,180 degrees as dataset to augment the biased dataset  </br>



![Class distribution of output variable](https://github.com/deep-kiran/Capstone/blob/master/output.png)


![Flow Chart](https://github.com/deep-kiran/Capstone/blob/master/Screenshot_20180903-210040-01.jpeg)

These are the snapshots of prototype. The input taken is a Fundus Image
![Snapshot](https://github.com/deep-kiran/Capstone/blob/master/Screenshot_20180903-214711-01.jpeg)
