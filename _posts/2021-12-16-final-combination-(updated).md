---
layout: post
title: "Rice Leaf Disease"
subtitle: "Image Classification to recognize which disease the rice leaves belongs to"
background: '/img/posts/final-combination-(updated)/bg-rice_leaf.jpg'
---

# Project Team ID = PTID-CDS-JUL21-1171 (Members - Diana, Hema, Pavithra and Sophiya)
# Project ID = PRCP-1001-RiceLeaf (Rice Leaf Disease using CNN)
___

### ------ Preliminary &rarr; Identify The Business Case ------

![png](/img/posts/final-combination-(updated)/rice2.png) <br/>
[Image Credit](https://shorturl.at/nuO67) <br/>
> - **Dataset Description:** The dataset contains images of rice leaf. The images are grouped into 3 classes based on the type of disease (a.k.a Bacterial leaf blight, Brown spot, and Leaf smut). There are 40 images in each class. The format of all images are jpg. <br/>
> - **Industry Field:** Agri/Forest. <br/>
> - **Decision:** Because we are dealing with images classification case, the most felicitous method to solve this case is by applying <u>neural network.

![png](/img/posts/final-combination-(updated)/1 Project Roadmap.png)

___

![png](/img/posts/final-combination-(updated)/Phase 1.png)

### -) Import Libraries


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import pandas as pd
import mplcyberpunk
import seaborn as sns
```


```python
#setting some arguments > pre-requisite
img_sz = 224
btc_sz = 32
channels = 3
```

### -) Loading Image Dataset from Directory


```python
dataset = tf.keras.utils.image_dataset_from_directory(
    "images",
    shuffle = True,
    image_size = (img_sz, img_sz),
    batch_size = btc_sz
)
class_names = dataset.class_names
class_names
```

    Found 119 files belonging to 3 classes.
    




    ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']



**``Interpretation:``** We know that 119 images (with proportion 40:40:39) for 3 classes are very less to the size that is required to train and build sophisticated models. This can indicate several consequences such as: ``very low accuracy, higher probabilities of overfitting, higher probabilities of misclassification.`` <br/><br/>
**``Decision:``** So before move forward into the modeling phase, there are several stages hence that the modeling for prediction can run smoothly as expected. The details are below as follows &darr; <br/>

___

![png](/img/posts/final-combination-(updated)/Phase 2.png)
### -) Create Data Augmentation Strategy


```python
#setting the augmented images into the associated folders
blb_dir = os.path.join('images', 'Bacterial leaf blight')  # directory with our class pictures
bs_dir = os.path.join('images', 'Brown spot')
ls_dir = os.path.join('images', 'Leaf smut')
```


```python
#create data augmentation
datagen = ImageDataGenerator(rotation_range=40, 
                         width_shift_range = 0.2, 
                         height_shift_range = 0.2,  
                         shear_range=0.2, 
                         zoom_range=0.2, 
                         horizontal_flip = True, 
                         fill_mode = 'nearest', 
                         data_format='channels_last', 
                         brightness_range=[0.5, 1.5])
```


```python
#augmenting data for the bacterial leaf blight and storing it into the respective sub-folder

i = 0

for batch in datagen.flow_from_directory(
    directory = 'images', #the directory name
    classes =  ['Bacterial leaf blight'],
    batch_size = btc_sz,  
    target_size = (img_sz, img_sz),
    color_mode = 'rgb', 
    save_to_dir = 'aug/blb_dir', 
    save_prefix = 'aug', 
    save_format='jpg'):
    
  i += 1
  if i>40:
    break
```

    Found 40 images belonging to 1 classes.
    


```python
#augmenting data for the Brown_spot and storing it into the respective sub-folder

i = 0

for batch in datagen.flow_from_directory(
    directory = 'images', #the directory name
    classes =  ['Brown spot'],
    batch_size = btc_sz,  
    target_size = (img_sz, img_sz),
    color_mode = 'rgb', 
    save_to_dir = 'aug/bs_dir', 
    save_prefix = 'aug', 
    save_format='jpg'):
    
  i += 1
  if i>40:
    break
```

    Found 40 images belonging to 1 classes.
    


```python
#augmenting data for leaf smut and storing it into the respective sub-folder

i = 0

for batch in datagen.flow_from_directory(
    directory = 'images', #the directory name
    classes =  ['Leaf smut'],
    batch_size = btc_sz,  
    target_size = (img_sz, img_sz),
    color_mode = 'rgb', 
    save_to_dir = 'aug/ls_dir', 
    save_prefix = 'aug', 
    save_format='jpg'):
    
  i += 1
  if i>40:
    break
```

    Found 39 images belonging to 1 classes.
    

### -) Loading Augmented Image


```python
# import the entire augmented dataset from directory
augmented_set = tf.keras.utils.image_dataset_from_directory(
    'aug',
    shuffle = True,
    image_size = (img_sz, img_sz),
    batch_size = btc_sz
)
```

    Found 2476 files belonging to 3 classes.
    

**``Summary:``** Augmented data from 119 images to 2476, which should be enough for classification problem, overdoing the image augmentation will overfit the model.


```python
len(augmented_set)
```




    78



### -) Splitting The Data &rarr; Using Tensorflow Input Pipeline


```python
#create function to splitting the data
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle = True, shuffle_size = 10000):
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
```


```python
train_ds, val_ds, test_ds = get_dataset_partitions_tf(augmented_set)
```


```python
print(len(train_ds))
print(len(val_ds))
print(len(test_ds))
```

    62
    7
    9
    

**``Interpretation:``** Every one batch involving 78 data (62 data for training, 7 for validation, and 9 for testing). Last batch = 32. So the total is 2476.

### -) Prefetch & Caching The Image Dataset
- **Illustration**
![png](/img/posts/final-combination-(updated)/prefetch & cache.png)
> Source belong to: [Tensorflow Guide](https://www.tensorflow.org/guide/data_performance#caching) <br/><br/>
- **``Conclusion:``** by applying prefetch and caching, it will improves the performance of the pipeline.


```python
#caching & prefetch (read image from the disk, & for the next iteration when we need the same image it will keep that image in the memory) => it will improves the performance of the pipeline
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
```

___

![png](/img/posts/final-combination-(updated)/Phase 3.png)
### -) Build CNN Model Architecture


```python
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(img_sz, img_sz),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])
```


```python
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2,),
])
```


```python
input_shape = (btc_sz, img_sz, img_sz, channels)
n_classes = 3

mymodel = models.Sequential([
    resize_and_rescale, #the first layer
    data_augmentation,
    layers.Conv2D(32,(3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(), # this converts our 3D feature maps to 1D feature vectors
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation = 'softmax'), #softmax will normalize the probability of classes
])

mymodel.build(input_shape=input_shape)
```

### -) Training The Model


```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# This callback will stop the training when there is no improvement in the 'loss' for three consecutive epochs.
modell = mymodel
modell.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)
history5 = modell.fit(
    train_ds, 
    epochs = 50,
    verbose=1,
    validation_data=val_ds,
    callbacks=[callback]
)
len(history5.history['loss'])  #
```

    Epoch 1/50
    62/62 [==============================] - 251s 3s/step - loss: 1.1023 - accuracy: 0.3304 - val_loss: 1.0989 - val_accuracy: 0.3393
    Epoch 2/50
    62/62 [==============================] - 215s 3s/step - loss: 1.0930 - accuracy: 0.3707 - val_loss: 1.1403 - val_accuracy: 0.3750
    Epoch 3/50
    62/62 [==============================] - 250s 4s/step - loss: 1.0703 - accuracy: 0.4129 - val_loss: 0.9850 - val_accuracy: 0.5179
    Epoch 4/50
    62/62 [==============================] - 272s 4s/step - loss: 0.9903 - accuracy: 0.5224 - val_loss: 0.9365 - val_accuracy: 0.5536
    Epoch 5/50
    62/62 [==============================] - 363s 6s/step - loss: 0.9505 - accuracy: 0.5692 - val_loss: 0.8994 - val_accuracy: 0.5357
    Epoch 6/50
    62/62 [==============================] - 343s 6s/step - loss: 0.9048 - accuracy: 0.5672 - val_loss: 0.8170 - val_accuracy: 0.6116
    Epoch 7/50
    62/62 [==============================] - 340s 5s/step - loss: 0.8948 - accuracy: 0.5626 - val_loss: 0.8195 - val_accuracy: 0.6116
    Epoch 8/50
    62/62 [==============================] - 341s 5s/step - loss: 0.8268 - accuracy: 0.6115 - val_loss: 0.7871 - val_accuracy: 0.6116
    Epoch 9/50
    62/62 [==============================] - 347s 6s/step - loss: 0.8171 - accuracy: 0.6064 - val_loss: 0.7686 - val_accuracy: 0.6696
    Epoch 10/50
    62/62 [==============================] - 285s 5s/step - loss: 0.8172 - accuracy: 0.6079 - val_loss: 0.6929 - val_accuracy: 0.6562
    Epoch 11/50
    62/62 [==============================] - 223s 4s/step - loss: 0.7283 - accuracy: 0.6609 - val_loss: 0.6760 - val_accuracy: 0.6964
    Epoch 12/50
    62/62 [==============================] - 216s 3s/step - loss: 0.6589 - accuracy: 0.6950 - val_loss: 0.5757 - val_accuracy: 0.7366
    Epoch 13/50
    62/62 [==============================] - 221s 4s/step - loss: 0.5760 - accuracy: 0.7546 - val_loss: 0.6767 - val_accuracy: 0.7098
    Epoch 14/50
    62/62 [==============================] - 216s 3s/step - loss: 0.5348 - accuracy: 0.7775 - val_loss: 0.4017 - val_accuracy: 0.8482
    Epoch 15/50
    62/62 [==============================] - 216s 3s/step - loss: 0.4547 - accuracy: 0.8192 - val_loss: 0.4389 - val_accuracy: 0.8348
    Epoch 16/50
    62/62 [==============================] - 211s 3s/step - loss: 0.4415 - accuracy: 0.8203 - val_loss: 0.3272 - val_accuracy: 0.8616
    Epoch 17/50
    62/62 [==============================] - 211s 3s/step - loss: 0.3718 - accuracy: 0.8467 - val_loss: 0.5266 - val_accuracy: 0.7679
    Epoch 18/50
    62/62 [==============================] - 201s 3s/step - loss: 0.3450 - accuracy: 0.8600 - val_loss: 0.2991 - val_accuracy: 0.8929
    Epoch 19/50
    62/62 [==============================] - 209s 3s/step - loss: 0.3234 - accuracy: 0.8676 - val_loss: 0.4110 - val_accuracy: 0.8214
    Epoch 20/50
    62/62 [==============================] - 210s 3s/step - loss: 0.3707 - accuracy: 0.8518 - val_loss: 0.4062 - val_accuracy: 0.8080
    Epoch 21/50
    62/62 [==============================] - 210s 3s/step - loss: 0.3276 - accuracy: 0.8702 - val_loss: 0.3121 - val_accuracy: 0.8661
    Epoch 22/50
    62/62 [==============================] - 208s 3s/step - loss: 0.2573 - accuracy: 0.8987 - val_loss: 0.3492 - val_accuracy: 0.8571
    Epoch 23/50
    62/62 [==============================] - 211s 3s/step - loss: 0.3557 - accuracy: 0.8473 - val_loss: 0.3633 - val_accuracy: 0.8795
    Epoch 24/50
    62/62 [==============================] - 212s 3s/step - loss: 0.3085 - accuracy: 0.8798 - val_loss: 0.2974 - val_accuracy: 0.8973
    Epoch 25/50
    62/62 [==============================] - 211s 3s/step - loss: 0.2695 - accuracy: 0.8926 - val_loss: 0.2842 - val_accuracy: 0.9018
    




    25



**``Interpretation:``** Even though we set epochs = 50, it will stop automatically at 25 epochs. The callback function plays a very important role in finding the optimum value of epochs in the model that has been created. <br/>
**``Decision:``** Print the summary of our model.


```python
#summary of cnn model architecture
modell.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    sequential (Sequential)      (None, 224, 224, 3)       0         
    _________________________________________________________________
    sequential_1 (Sequential)    (None, 224, 224, 3)       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 222, 222, 32)      896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 111, 111, 32)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 109, 109, 64)      18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 54, 54, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 52, 52, 64)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 26, 26, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 10, 10, 64)        36928     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 3, 3, 64)          36928     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 1, 1, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 64)                0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                4160      
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 195       
    =================================================================
    Total params: 171,459
    Trainable params: 171,459
    Non-trainable params: 0
    _________________________________________________________________
    

**``Interpretation:``** Summary above shows us we've been successfully train our params, it runs smoothly. So the probability of success to dealing with prediction is higher, as we don't have "non-trainable params". <br/>
**``Decision:``** Move forward into evaluating the model.

### -) Evaluate The Model


```python
print('The parameters inside our model: ', history5.params)
print('The keys inside our model: ', history5.history.keys())
```

    The parameters inside our model:  {'verbose': 1, 'epochs': 50, 'steps': 62}
    The keys inside our model:  dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    


```python
scores = modell.evaluate(test_ds)
```

    9/9 [==============================] - 33s 535ms/step - loss: 0.2622 - accuracy: 0.8924
    


```python
print('Train accuracy: ', round(history5.history['accuracy'][-1]*100,2), '%')
print('Test accuracy: ', round(scores[-1]*100, 2), '%')
```

    Train accuracy:  89.26 %
    Test accuracy:  89.24 %
    

**``Interpretation:``** Result above are not much different (still in the same range / 89%), that's indicates our CNN model architecture is pretty much suitable to solve this case. <br/> **``Decision:``** To ascertain, let's display a random image to test our prediction is suitable to solve this classification. It is shown as following below &darr;


```python
#display a random image to test the prediction
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print('First image to predict')
    plt.imshow(first_image)
    print('Actual label:', class_names[first_label])
    
    batch_prediction = mymodel.predict(images_batch)
    print('Predicted label:',class_names[np.argmax(batch_prediction[0])])
```

    First image to predict
    Actual label: Bacterial leaf blight
    Predicted label: Bacterial leaf blight
    


    
![png](/img/posts/final-combination-(updated)/output_42_1.png)
    



```python
#export model to a file on disk >> using auto increment
mymodel_version = max([int(i) for i in os.listdir('./preview/mymodel') + [0]])+1
mymodel.save(f"./preview/mymodel/{mymodel_version}")
mymodel.save("CNN_RiceLeafDiseaseClassification.h5")
```

    INFO:tensorflow:Assets written to: ./preview/mymodel/2\assets
    

___

![png](/img/posts/final-combination-(updated)/Phase 4.png)
### -) Visualize The Result


```python
# Summary of the Accuracy scores for splitting dataset
model_ev = pd.DataFrame({'Set': ['Train','Test','Validation'], 'Accuracy (%)': [round(history5.history['accuracy'][-1]*100,2),
                    round(scores[-1]*100, 2),round(history5.history['val_accuracy'][-1]*100,2)]})
model_ev
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>89.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Test</td>
      <td>89.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Validation</td>
      <td>90.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_theme(style="darkgrid")
plt.figure(figsize=(15,7))
plt.title("Accuracy Comparison", size = 20, fontweight='bold')
plt.xlabel("[ Set ]", fontweight='bold', size = 15)
plt.ylim(85.5, 90.95, 0.005)
plt.xticks(rotation=0)
plt.ylabel("[ Accuracy (%) ]", fontweight='bold', size = 15)
plt.xticks(size = 14)
plt.yticks(size = 14)
ax = sns.barplot(x="Set", y="Accuracy (%)", data=model_ev,
                 palette="Blues_d")
ax = sns.pointplot(x="Set", y="Accuracy (%)", data=model_ev, capsize=.2, color='#bb3f3f', linestyles='--')
```


    
![png](/img/posts/final-combination-(updated)/output_47_0.png)
    


**``Interpretation:``** That's how it looks like, the differences in accuracy between the 3 of them (< 1%/each) are not showing any warning sign to do further treatment. <br/>
**``Decision:``** Move forward to visualize the accuracy and loss for the training and validation set.


```python
acc = history5.history['accuracy']
val_acc = history5.history['val_accuracy']

loss = history5.history['loss']
val_loss = history5.history['val_loss']

#plotting training and validation (with respect to accuracy and loss)
plt.style.use('cyberpunk')
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.plot(range(25), acc, label='Training Accuracy', marker = 'o')
plt.plot(range(25), val_acc, label='Validation Accuracy', marker = 'P')
plt.xlabel('[ number of epochs ]', size = 13)
plt.ylabel('[ accuracy ]', size = 13)
plt.legend(loc = 'lower right')
plt.title('Training & Validation Accuracy', fontweight = 'bold', size = 14)
plt.xticks(size = 14)
plt.yticks(size = 14)
mplcyberpunk.add_glow_effects()

plt.subplot(1, 2, 2)
plt.plot(range(25), loss, label='Training Loss', marker = 'o')
plt.plot(range(25), val_loss, label='Validation Loss', marker = 'P')
plt.xlabel('[ number of epochs ]', size = 13)
plt.ylabel('[ loss ]', size = 13)
plt.legend(loc = 'upper right')
plt.title('Training & Validation Loss', fontweight = 'bold', size = 14)
plt.xticks(size = 14)
plt.yticks(size = 14)
mplcyberpunk.add_glow_effects()
plt.show()
```


    
![png](/img/posts/final-combination-(updated)/output_49_0.png)
    


**``Interpretation:``** According to the graphs above, as the number of epochs is increasing, the CNN model accuracy that we build is also increasing (getting closer to 100%). That indicates our model performs better and it's absolutely suitable to solve this case. Whereas, 'loss' is the result of a bad prediction. A 'loss' is a number indicating how bad the model's prediction was. If the model's prediction is perfect, the 'loss' is 0. On the right side, we can clearly see as the number of epochs are increasing, the more our 'loss' is much closer to 0. <br/>
**``Decision:``** Move forward to the final stage (Visualize The Prediction).


```python
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy()) #convert image into array
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = mymodel.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    return predicted_class, confidence
```


```python
#Visualize the prediction
plt.figure(figsize = (15,15))
plt.suptitle('Display of Prediction Rice Leaf Disease Image', fontweight = 'bold', size = 25)
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        
        predicted_class, confidence = predict(modell, images[i].numpy())
        actual_class = class_names[labels[i]]
        
        plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
```


    
![png](/img/posts/final-combination-(updated)/output_52_0.png)
    


### -) Project Risks

1. We have used data augmentation technique to increase the size of the dataset from 119 to over 2476 images, which may have oversimplified and overfit the model to the same dataset and may not be a representative of the entire population images of these diseases.
2. The project assumes that the dataset used for modeling in this project is a representative of the population dataset else the models may not provide the accuracies that are shown here.

### -) Reccomendations

1. CNN model above is a good modeling techinque for this project with good accuracy given the limited dataset.
2. Model can also be further improved with the hyperparameter tuning of the CNN architecture but since the accuracy is within the acceptable range we limit the scope and decided to stop further analysis.

## _Additional Context &rarr; CNN Model Architecture Diagram_


```python
from IPython.core.display import display, HTML
display(HTML("""<a href="shorturl.at/suLS8">CNN Model Architecture Diagram</a>"""))
from IPython.display import Image
Image(filename='CNN_RiceLeafDiseaseClassification.h5.png')
```


<a href="shorturl.at/suLS8">CNN Model Architecture Diagram</a>





    
![png](/img/posts/final-combination-(updated)/output_58_1.png)
    


