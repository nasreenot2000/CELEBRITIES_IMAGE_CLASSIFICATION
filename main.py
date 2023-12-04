import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from keras.utils import to_categorical

image_dir='cropped'
lional_messi_images=os.listdir(image_dir+ '/lionel_messi')
maria_sharapova_images=os.listdir(image_dir+ '/maria_sharapova')
roger_federer_images=os.listdir(image_dir+'/roger_federer')
serena_williams_images=os.listdir(image_dir+'/serena_williams')
virat_kohli_images=os.listdir(image_dir+'/virat_kohli')
print("--------------------------------------\n")

print('The length of lional_messi_images is',len(lional_messi_images))
print('The length of maria_sharapova_images is',len(maria_sharapova_images))
print('The length of roger_federer_images is',len(roger_federer_images))
print('The length of serena_williams_images is',len(serena_williams_images))
print('The length of virat_kohli_images is',len(virat_kohli_images))

print("--------------------------------------\n")


dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(lional_messi_images),desc="Lional_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_sharapova_images),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i ,image_name in tqdm(enumerate(roger_federer_images),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)

for i ,image_name in tqdm(enumerate(serena_williams_images),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)

for i ,image_name in tqdm(enumerate(virat_kohli_images),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)
        
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

# x_train=x_train.astype('float')/255
# x_test=x_test.astype('float')/255 

# Same step above is implemented using tensorflow functions.

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")
img_size = (128, 128, 3)  # Adjust this according to your actual image size
num_classes = 5  # Change this to the number of classes in your problem

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=img_size),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
])



'''data_augmentation = tf.keras.Sequential(
   [
     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=img_siz),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
     tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
   ]
 )


model=tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])'''


print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
y_train_one_hot = to_categorical(y_train, num_classes=5)
history=model.fit(x_train,y_train_one_hot,epochs=100,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")



y_test_one_hot = to_categorical(y_test, num_classes=5)

print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test_one_hot)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
print('classification Report\n',classification_report(y_test_one_hot,y_pred))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")


