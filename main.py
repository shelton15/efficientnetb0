import numpy as np
from sklearn.metrics import f1_score, fbeta_score
from efficientnet.tfkeras import EfficientNetB0
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Define dataset paths
medieval_path = "path/to/medieval/dataset"
weibo_path = "path/to/weibo/dataset"
casia_path = "path/to/casia/dataset"

# Define dataset transforms
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Define datasets
train_medieval_dataset = train_datagen.flow_from_directory(
    medieval_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_medieval_dataset = val_datagen.flow_from_directory(
    medieval_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

train_weibo_dataset = train_datagen.flow_from_directory(
    weibo_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_weibo_dataset = val_datagen.flow_from_directory(
    weibo_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

train_casia_dataset = train_datagen.flow_from_directory(
    casia_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_casia_dataset = val_datagen.flow_from_directory(
    casia_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Define model
num_classes = 2  # Assuming binary classification
model = EfficientNetB0(include_top=True, weights=None, classes=num_classes)

# Define loss function and optimizer
criterion = SparseCategoricalCrossentropy(from_logits=True)
optimizer = Adam(lr=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

# Train the model on Medieval, Weibo, and CASIA datasets
num_epochs = 10
for epoch in range(num_epochs):
    for i, ((medieval_images, _), medieval_labels) in enumerate(train_medieval_dataset):
        _, weibo_data = next(train_weibo_dataset)
        _, casia_data = next(train_casia_dataset)
        images = np.concatenate((medieval_images, weibo_data[0], casia_data[0]), axis=0)
        labels = np.concatenate((medieval_labels, weibo_data[1], casia_data[1]), axis=0)
        images, labels = shuffle(images, labels)

        model.train_on_batch(images, labels)

# Evaluate the model on Medieval, Weibo, and CASIA datasets
medieval_preds = []
medieval_labels = []
for images, labels in val_medieval_dataset:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    medieval_preds += preds.tolist()
    medieval_labels += labels.tolist()

medieval_f1 = f1_score(medieval_labels, medieval_preds, average='weighted')
medieval_f2 = fbeta_score(medieval_labels, medieval_preds, beta=2, average='weighted')

weibo_preds = []
weibo_labels = []
for images, labels in val_weibo_dataset:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    weibo_preds += preds.tolist()
    weibo_labels += labels.tolist()

weibo_f1 = f1_score(weibo_labels, weibo_preds, average='weighted')
weibo_f2 = fbeta_score(weibo_labels, weibo_preds, beta=2, average='weighted')

casia_preds = []
casia_labels = []
for images, labels in val_casia_dataset:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    casia_preds += preds.tolist()
    casia_labels += labels.tolist()

casia_f1 = f1_score(casia_labels, casia_preds, average='weighted')
casia_f2 = fbeta_score(casia_labels, casia_preds, beta=2, average='weighted')

print("Medieval F1 score:", medieval_f1)
print("Medieval F2 score:", medieval_f2)
print("WeiboF1 score:", weibo_f1)
print("Weibo F2 score:", weibo_f2)
print("CASIA F1 score:", casia_f1)
print("CASIA F2 score:", casia_f2)
