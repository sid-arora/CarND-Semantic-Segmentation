[image1]: ./runs/1528867024.4458013/um_000005.png "1"
[image2]: ./runs/1528867024.4458013/umm_000005.png "2"
[image3]: ./runs/1528867024.4458013/uu_000006.png "3"
[image4]: ./runs/1528867024.4458013/uu_000071.png "4"
[image5]: ./runs/1528867024.4458013/uu_000052.png "5"
[image6]: ./runs/1528867024.4458013/umm_000043.png "6"

# Semantic Segmentation
### Introduction
In this project, I labelled the pixels of a road in images using a Fully Convolutional Network (FCN) based on the VGG-16 image classifier architecture

Used pre-trained VGG-16 network and converted it to a fully convolutional network by converting the final fully connected layer to a 1x1 convolution. Two classes were used : road and not-road. Performance gains using : skip connections, 1x1 convolutions and sampling.
I then ran the Model training with Adam Optimizer and inference using AWS GPU.

### Run:

```
python main.py
```
The command above creates an output runs folder as in the repository. 

#### Run Output:

```
ubuntu@ip-172-31-19-21:~/CarND-Semantic-Segmentation$ python main.py
/home/ubuntu/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
TensorFlow Version: 1.9.0
main.py:15: UserWarning: No GPU found. Please use a GPU to train your neural network.
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
Tests Passed
Tests Passed
WARNING:tensorflow:From main.py:118: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See @{tf.nn.softmax_cross_entropy_with_logits_v2}.

Tests Passed
Tests Passed
Tests Passed
2018-08-01 06:43:22.848911: W tensorflow/core/graph/graph_constructor.cc:1248] Importing a graph with a lower producer version 21 into an existing graph with producer version 26. Shape inference will have run different parts of the graph with different producer versions.
Training Neural Network


Epoch No. 1 
/home/ubuntu/anaconda3/lib/python3.6/site-packages/scipy/misc/pilutil.py:482: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
  if issubdtype(ts, int):
/home/ubuntu/anaconda3/lib/python3.6/site-packages/scipy/misc/pilutil.py:485: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  elif issubdtype(type(size), float):
Training Loss: 1.750
Training Loss: 5.308
Training Loss: 0.891
Training Loss: 0.708
Training Loss: 0.755
Training Loss: 0.751
Training Loss: 0.710
Training Loss: 0.677
Training Loss: 0.715
Training Loss: 0.717
Training Loss: 0.675
Training Loss: 0.646
Training Loss: 0.630
Training Loss: 0.604
Training Loss: 0.581
Training Loss: 0.559
Training Loss: 0.552
Training Loss: 0.499
Training Loss: 0.515
Training Loss: 0.463
Training Loss: 0.437
Training Loss: 0.403
Training Loss: 0.347
Training Loss: 0.375
Training Loss: 0.479
Training Loss: 0.480
Training Loss: 0.403
Training Loss: 0.283
Training Loss: 0.562
Training Loss: 0.388
Training Loss: 0.347
Training Loss: 0.390
Training Loss: 0.387
Training Loss: 0.347
Training Loss: 0.371
Training Loss: 0.348
Training Loss: 0.354
Training Loss: 0.280
Training Loss: 0.255
Training Loss: 0.280
Training Loss: 0.229
Training Loss: 0.266
Training Loss: 0.269
Training Loss: 0.281
Training Loss: 0.235
Training Loss: 0.253
Training Loss: 0.253
Training Loss: 0.264
Training Loss: 0.210
Training Loss: 0.202
Training Loss: 0.153
Training Loss: 0.234
Training Loss: 0.185
Training Loss: 0.204
Training Loss: 0.184
Training Loss: 0.213
Training Loss: 0.135
Training Loss: 0.183

Epoch No. 2 
Training Loss: 0.342
Training Loss: 0.159
Training Loss: 0.209
Training Loss: 0.225
Training Loss: 0.217
Training Loss: 0.220
Training Loss: 0.242
Training Loss: 0.160
Training Loss: 0.169
Training Loss: 0.299
Training Loss: 0.229
Training Loss: 0.180
Training Loss: 0.232
Training Loss: 0.178
Training Loss: 0.197
Training Loss: 0.192
Training Loss: 0.141
Training Loss: 0.206
Training Loss: 0.144
Training Loss: 0.187
Training Loss: 0.161
Training Loss: 0.209
Training Loss: 0.198
Training Loss: 0.166
Training Loss: 0.140
Training Loss: 0.207
Training Loss: 0.178
Training Loss: 0.163
Training Loss: 0.205
Training Loss: 0.157
Training Loss: 0.151
Training Loss: 0.140
Training Loss: 0.178
Training Loss: 0.156
Training Loss: 0.151
Training Loss: 0.198
Training Loss: 0.150
Training Loss: 0.216
Training Loss: 0.172
Training Loss: 0.161
Training Loss: 0.163
Training Loss: 0.211
Training Loss: 0.150
Training Loss: 0.162
Training Loss: 0.153
Training Loss: 0.155
Training Loss: 0.101
Training Loss: 0.202
Training Loss: 0.159
Training Loss: 0.161
Training Loss: 0.133
Training Loss: 0.201
Training Loss: 0.161
Training Loss: 0.208
Training Loss: 0.189
Training Loss: 0.153
Training Loss: 0.133
Training Loss: 0.160

Epoch No. 3 
Training Loss: 0.187
Training Loss: 0.156
Training Loss: 0.194
Training Loss: 0.196
Training Loss: 0.234
Training Loss: 0.180
Training Loss: 0.124
Training Loss: 0.136
Training Loss: 0.149
Training Loss: 0.193
Training Loss: 0.133
Training Loss: 0.109
Training Loss: 0.115
Training Loss: 0.109
Training Loss: 0.168
Training Loss: 0.149
Training Loss: 0.136
Training Loss: 0.121
Training Loss: 0.187
Training Loss: 0.190
Training Loss: 0.118
Training Loss: 0.168
Training Loss: 0.112
Training Loss: 0.104
Training Loss: 0.168
Training Loss: 0.165
Training Loss: 0.126
Training Loss: 0.167
Training Loss: 0.195
Training Loss: 0.110
Training Loss: 0.145
Training Loss: 0.140
Training Loss: 0.146
Training Loss: 0.162
Training Loss: 0.208
Training Loss: 0.129
Training Loss: 0.195
Training Loss: 0.154
Training Loss: 0.118
Training Loss: 0.212
.......................
```
#### Output Images:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

### Rubric:

 - Does the project load the pretrained vgg model? YES
 - Does the project learn the correct features from the images? YES
 - Does the project optimize the neural network? YES
 - Does the project train the neural network? YES
 - Does the project train the model correctly? YES
 - Does the project use reasonable hyperparameters? YES
      - keep_prob: 0.5
      - epochs : 50
      - batch size : 5
      - learning rate : 0.5
 - Does the project correctly label the road? YES

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
