# QR Code Detection and Image Classification

  

## Step 1. Warp QR Code Perespective

  

To achive correct perspective of our qrcode aka getting bird-view angle,

  

1. first we apply morphological closing and dialating operators to fill all the gaps between black holes. In this way we get some black blocks and we assume the biggest block is the qrcode.

2. We get the largest block by finding the largest contour of the filled image by finding the edges of image.

3. we find an approximated polygon that surrounds the selected contour, if the polygon has more than four vertexes we find a rectangle that bound the polygon. 

4. Then we find the closest points of the qrcode to each vertex and warp the perspective of the qrcode to getting a square look of the image.

## Step 2. Read QR Code Data

We use `pyzbar` library to read the data within the qrcode and display it.


## Step 3. Fine-tune Classifier

To classify the specified image inside the qrcode we fine-tune a [YOLOv11n-cls](https://docs.ultralytics.com/tasks/classify/)  model which is the smallest Yolo-based classifier. 

We gather a custom dataset of the 3 specified classes using [Roboflow](https://app.roboflow.com/) which is publicly available via the following [link](https://universe.roboflow.com/alidatasets/rsi-test).

You can train the model using `rsi_fine_tune_yolo11_classifier.ipynb` file. (**Fine-tuning can be done in only 5 epochs with accuracy of 100% in less than a minute!!!**)

The trained model exists inside the models folder.
## Step 4. Classify

After reading qrcode data, we classify the specified image using YOLO model and report the class with highest probability. Due to the fact that our classes are highly separable our tiny model can classify them with certainty and without error.


## How to run ?

Install requirements using (generated using `pipreqs`): 

```bash
pip install -r requirements.txt
```

Run the project:
```bash
python main.py --debug # to enable display process of each step
python main.py --help # to find out possible arguments
```