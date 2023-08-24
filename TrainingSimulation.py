# print('Setting UP')
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_selection import train_test_split
#### Step:1
path = 'Data'
data = importDataInfo(path)

#### Step:2
data = balanceData(data, display=False)

#### Step:3
imagesPath, steerings = loadData(path,data)

#### Step:4 [Split for Training and Validation]
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### Step:5 [Data Augmentation]

#### Step:6 [Preprocessing]

#### Step:7 [Batch Generator]


#### Step:8 [Create the Model]
model = createModel()
model.summary()

#### Step:9

history = model.fit(batchGen(xTrain,yTrain,10,1),steps_per_epoch=20,epochs=2,
          validation_data=batchGen(xVal,yVal,10,0),validation_steps=20)
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()