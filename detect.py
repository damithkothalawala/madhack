#!/usr/bin/env python
# coding: utf-8

#get_ipython().run_line_magic('matplotlib', 'inline')
import base64
import json
import io
import os
import shutil
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

#print('Using PyTorch version', torch.__version__)


# # Preparing Training and Test Sets

# In[19]:


class_names = ['normal', 'viral', 'covid']
root_dir = 'COVID-19 Radiography Database'
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 30)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)


# # Creating Custom Dataset

# In[93]:

class ChestXRayDatasetX(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            #print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        self.class_names = ['input']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


# # Image Transformations

# In[21]:


test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# # Prepare DataLoader

# In[22]:



# In[119]:


test_dirs = {
    'input': 'test/'

}

test_dataset = ChestXRayDatasetX(test_dirs, test_transform)


# In[118]:



dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)



# # Data Visualization

# In[98]:




def show_images(images, labels, preds):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 1, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        #plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.xlabel('Sadrushya Makato Result')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    #plt.savefig('out.png')
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    print(json.dumps(base64.b64encode(pic_IObytes.read()).decode("utf-8"))) # Print Base64 Image
    #print(pic_hash)
# In[117]:





#resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18 = torch.load('MadHacktrainedCovid19')
resnet18.fc = nn.Linear(resnet18.fc.in_features, 3)
resnet18= resnet18.to('cpu')
resnet18.load_state_dict(torch.load('trained',map_location='cpu'))


# In[29]:


resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)


# In[104]:


def show_preds():
    resnet18.eval()
    images, labels = next(iter(dl_test))
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds)


show_preds()

