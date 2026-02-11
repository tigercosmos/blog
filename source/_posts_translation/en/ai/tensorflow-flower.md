---
title: "Building a Flower Classification App with TensorFlow"
date: 2017-07-28 00:42:09
tags: [ai, tensorflow]
lang: en
translation_key: tensorflow-flower
---

# Preface

Recently, terms like machine learning, artificial intelligence, and deep learning have become extremely popular. In this post, I use deep learning to build a simple Android app that can recognize different types of flowers. Part 1 trains an image model, which you can already use for classification on your computer. Part 2 puts the trained model into an Android project so you can do recognition on a phone.

<!-- more --> 

# Part 1
## Environment Setup

### Anaconda
Anaconda lets you create virtual environments to run Python. The benefit is that you can create many different environments for testing, and when using Anaconda, `pip` does not require root privileges.
```
$ wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
$ bash Anaconda3-4.4.0-Linux-x86_64.sh
```

### Python & TensorFlow
Create a virtual environment. Since I used Python 2.7 with TensorFlow 1.1, I named the environment `py2t1.1`.
```
$ conda create -n py2t1.1 python=2.7
```
After installing, enter the virtual environment. To activate, use `activate`:
```
 $ source activate py2t1.1
 (py2t1.1)$  # it will look like this
```
To deactivate, use `deactivate`:
```
 $ source deactivate
```

Next, install `tensorflow`. Here I use TensorFlow 1.1 (Python 2.7, CPU). If you need other versions, you can find them [here](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package).
```
 (py2t1.1)$ pip install --ignore-installed --upgrade \
 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
```

Now the environment is ready.

## Training the Model

### Preparation
At this point, you can be inside the virtual environment or not. You only need to enter the environment when you run TensorFlow with Python.

Create a folder:
```
$ mkdir flower_classification
$ cd flower_classification
```

Download the flower photos:
```
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```

Download `retrain.py`. It uses an `Inception v3` image recognition model and adds a new final layer, then retrains the model to classify flowers.
```
$ curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```

### Start Training
First, activate the environment:
```
$ source activate py2t1.1
```

```
$ python retrain.py \
  --bottleneck_dir=bottlenecks \
  --model_dir=inception \
  --learning_rate=0.5 \
  --summaries_dir=training_summaries/long \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=flower_photos
```

After training, it generates `retrained_graph.pb` and `retrained_labels.txt`. You can use these two files to classify new photos.

### Classifying Photos
Download `label_image.py`, which is the script for classification:
```
$ curl -L https://goo.gl/3lTKZs > label_image.py
```

Run classification:
```
$ python label_image.py 要辨識的花朵照片.jpeg
```

The output will look like this:
```
daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)
```

# Part 2
## Preparation
`tensorflow-for-poets-2` is a sample Android app project from Google Codelab.
```
$ cd ~/
$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
$ cd tensorflow-for-poets-2
```
Copy the entire flower classification folder `flower_classification` (trained earlier) into the project:
```
$ cp -r ~/flower_classification .
```

## Processing the Model
Optimize:
```
$ python -m tensorflow.python.tools.optimize_for_inference \
  --input=flower_classification/retrained_graph.pb \
  --output=flower_classification/optimized_graph.pb \
  --input_names="Cast" \
  --output_names="final_result"
```
Quantize:
```
$ python -m scripts.quantize_graph \
  --input=flower_classification/optimized_graph.pb \
  --output=flower_classification/rounded_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded
```

Put `rounded_graph.pb` and `retrained_labels.txt` into the `android/assets` folder:
```
$ cp flower_classification/rounded_graph.pb flower_classification/retrained_labels.txt android/assets/ 
```

## Installing Android Studio
https://developer.android.com/studio/index.html

## Opening the Android Project
Open Android Studio, choose “Open an existing Android Studio project”, and select the folder `tensorflow-for-poets-2/android`. Then it will ask whether to use Gradle—choose “ok”. After Gradle finishes syncing, you can build the project, and finally install the APK onto your phone.

## References
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#0

