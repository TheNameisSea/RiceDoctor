#_____Imports_____#
# Timm
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

# Settings
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from collections import defaultdict
import pandas as pd
import numpy as np
import os, shutil
import random
import gc
import cv2
from PIL import Image
import glob
gc.enable()
pd.set_option('display.max_columns', None)

# Augmentations
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# Deep Learning
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F

# UI
import streamlit as st

# Utils
import tempfile

# Seeds
RANDOM_SEED = 42

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything()

# Device
device = torch.device('cpu')

#_____Data_____#
# required datafile
data_dir = './input'        # image folder
models_dir = './models'     # model folder
test_df = pd.DataFrame(columns=['image_id', 'label'])

id2label = {0: 'Bạc lá',
            1: 'Đốm sọc lá',
            2: 'Bạc chuỳ lá',
            3: 'Cháy lúa',
            4: 'Đốm nâu',
            5: 'Chết lá non',
            6: 'Sương mai',
            7: 'Bọ gai',
            8: 'Bình thường',
            9: 'Vi-rút Tungro'}

# Params
params = {
    'model': 'efficientnet_b3',
    'pretrained': False,
    'inp_channels': 3,
    'im_size': 300,
    'device': device,
    'batch_size': 85,
    'num_workers' : 0,
    'out_features': 10,
    'dropout': 0.2,
    'num_fold': 10,
    'debug': False,
}

# Transform
def get_test_transforms(DIM = params['im_size']):
    return albumentations.Compose(
        [
          albumentations.Resize(DIM,DIM),
          albumentations.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          ),
          ToTensorV2(p=1.0)
        ]
    )

# Dataset and Dataloader
class PaddyDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image

#_____Deep Learning_____#
# Neural Net
class PaddyNet(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['out_features'], inp_channels=params['inp_channels'],
                 pretrained=params['pretrained']):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
        out_channels = self.model.conv_stem.out_channels
        kernel_size = self.model.conv_stem.kernel_size
        stride = self.model.conv_stem.stride
        padding = self.model.conv_stem.padding
        bias = self.model.conv_stem.bias
        self.model.conv_stem = nn.Conv2d(inp_channels, out_channels,
                                          kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=bias)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.dropout = nn.Dropout(params['dropout'])
        self.fc = nn.Linear(n_features, out_features)
    
    def forward(self, image):
        embeddings = self.model(image)
        x = self.dropout(embeddings)
        output = self.fc(x)
        return output
    
# Utils
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Page cfgs
st.set_page_config(page_title="Rice Disease Classification", page_icon="🔬", layout="centered", initial_sidebar_state="expanded", menu_items=None) 

# Page Title
st.write("""
# Bác Sĩ Lúa
Chẩn đoán bệnh lúa dựa trên hình ảnh 
""")
         
st.sidebar.header('Tải ảnh lên để bắt đầu chẩn đoán')
uploaded_files = st.sidebar.file_uploader("Tải ảnh", accept_multiple_files=True)

res_col, img_col = st.columns([4, 2])

if uploaded_files is not None:
    try:
        if os.path.exists('./upload'):
            shutil.rmtree('./upload')
        os.mkdir('./upload')
        upload_path = "./upload"

        os.chdir(upload_path)
        for uploaded_file in uploaded_files:
            img = load_image(uploaded_file)
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        os.chdir('../')

        image_ids = os.listdir(upload_path)      # take name of all uploaded file
        if '.DS_Store' in image_ids:
            image_ids.remove('.DS_Store')
        # folder_len = len(image_ids)     # Take number of uploaded file

        # test_df = test_df[(test_df.index < folder_len)]     # remove images that could cause overflow
        test_df['image_id'] = image_ids     # add name of uploaded file to template
        paths = test_df.apply(lambda row: './upload/' + row['image_id'], axis=1)
        paths = paths.to_numpy()
        test_df['image_path'] = paths    # create filepaths



        # Prediction
        pred_cols = []

        for i, model_name in enumerate(glob.glob(models_dir + '/*.pth')):
            model = PaddyNet()
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))     # load model
            model = model.to(params['device'])
            model.eval()        # evaluate model for uses
            
            X_test = test_df['image_path']      # read filepaths

            # create a dataset of images and apply augmentation to the images
            test_dataset = PaddyDataset(
                images_filepaths=X_test.values,     
                transform = get_test_transforms()
            )
            
            # data loader is used to load data
            test_loader = DataLoader(
                test_dataset, batch_size=params['batch_size'],
                shuffle=False, num_workers=params['num_workers'],
                pin_memory=True
            )

            temp_preds = None
            with torch.no_grad():
                for images in test_loader:
                    images = images.to(params['device'], non_blocking=True)
                    predictions = model(images).softmax(dim=1).argmax(dim=1).to('cpu').numpy()      # apply softmax and find the highest possibility
                    
                    if temp_preds is None:
                        temp_preds = predictions
                    else:
                        temp_preds = np.hstack((temp_preds, predictions))

            test_df[f'model_{i}_preds'] = temp_preds
            pred_cols.append(f'model_{i}_preds')

        test_df['label'] = test_df[pred_cols].mode(axis=1)[0]
        test_df = test_df[['image_id', 'label']]
        label_col = test_df['label'].copy()
        test_df['label'] = test_df['label'].map(id2label)

        with res_col:
            st.write(test_df)
            for i in label_col.unique():
                if i == 0:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/14_benhbachla.htm")
                elif i == 1:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/26_phongtrusaubenh.htm")
                elif i == 2:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/14_benhbachla.htm")
                elif i == 3:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/26_phongtrusaubenh.htm")
                elif i == 4:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/26_phongtrusaubenh.htm")
                elif i == 5:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/01_sauducthan.htm")
                elif i == 6:
                    st.write("https://www.fao.org.vn/sau-benh/suong-mai/")
                elif i == 7:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/08_saugai.htm")
                elif i == 8:
                    st.write("")
                elif i == 9:
                    st.write("https://vaas.vn/kienthuc/Caylua/06/17_benhlunxoanla.htm")

                    

        with img_col:
            for img_path in os.listdir(upload_path):
                image = Image.open(os.path.join('./upload', img_path))
                st.image(image, use_column_width='auto')

        test_df.to_csv('./output/submission.csv', index=False)

    except:
        pass




