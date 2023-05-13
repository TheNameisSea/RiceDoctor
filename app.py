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
test_df = pd.DataFrame()

id2label = {0: 'Bạc lá',
            1: 'Đốm sọc lá',
            2: 'Bạc chuỳ lá',
            3: 'Cháy lúa',
            4: 'Đốm nâu',
            5: 'Sâu đục thân lúa',
            6: 'Sương mai',
            7: 'Bọ gai',
            8: 'Bình thường',
            9: 'Vi-rút Tungro',
            10: 'Không xác định'}

bacla = "Xử lý bệnh bạc lá:\n Sử dụng các thuốc trong danh mục thuốc BVTV được phép sử dụng ở Việt Nam có chứa các hoạt chất Bismerthiazol, Copper hydroxide, Oxolinic acid, Thiodiazole zinc, Thiodiazole copper,… để phun.\nGiai đoạn lúa đòng – trổ – chín, bà con cần theo dõi chặt chẽ diễn biến của thời tiết tiến hành phun trước và sau mưa giông bằng các thuốc bảo vệ thực vật có chứa hoạt chất nêu trên theo nguyên tắc 4 đúng và theo hướng dẫn sử dụng trên bao bì để ngăn chặn bệnh lây lan trên diện rộng."
domsocla = "Xử lý bệnh đốm sọc lá:\nKhông bón quá nhiều đạm, bón đạm muộn và kéo dài; Chú ý kết hợp giữa bón đạm với phân chuồng, lân, kali.\nKhi phát hiện thấy ruộng chớm bị bệnh cần giữ mực nước từ 3 - 5 cm, tuyệt đối không được bón các loại phân hóa học, phân bón qua lá và thuốc kích thích sinh trưởng.\n Sử dụng các loại thuốc hoá học để phun phòng trừ như: Totan 200WP, Starwiner 20WP, Novaba 68WP, Xanthomix 20WP, PN - Balacide 32 WP, Starner 20 WP,... pha và phun theo hướng dẫn trên vỏ bao bì." 
bacchuyla = "Xử lý bệnh bạc chuỳ lá:\nChọn giống lúa ít nhiễm bệnh, làm đất kỹ, giải độc hữu cơ, sạ thưa, bón phân cân đối NPK, bón đầy đủ canxi, silic, kali, không để nước ngập sâu kéo dài.\n Vào giai đoạn cuối đẻ nhánh, nuôi đòng nếu xuất hiện bệnh Cháy bìa lá do vi khuẩn có thể sử dụng Agrilife 100SL với liều 30ml/25 lít.\nVào giai đoạn lúa trổ đến chín: phun thuốc 25ml Agrilife 100SL +25ml Keviar 325SC/25 lít 2 lần: lúc lúa trổ lẹt xẹt và lúc lúa trổ đều."
chaylua = "Xử lý bệnh cháy lúa:\nTrước khi trồng vụ mới thì tiến hành cải tạo và vệ sinh sạch đồng ruộng để tránh mầm khuẩn gây bệnh.\nBón phân cân đối dựa trên màu lá lúa, chú ý giữ mực nước trong ruộng thích hợp từ 5-10cm so với cây lúa.\nKhi mới phát hiện bệnh, nên rút nước trong ruộng ra rồi tiến hành rải vôi với liều lượng 10 – 20kg/1.000m2.\nSử dụng thuốc đặc trị bệnh cháy lá lúa khi cần thiết, sử dụng theo hướng dẫn của nhà sản xuất trên bao bì."
domnau = "Xử lý bệnh đốm nâu:\nXử lý giống trước khi gieo bằng nước nóng 54 độ C hoặc bằng cách dùng một trong các loại thuốc trừ bệnh như Carban 50SC, Vicarben 50HP... pha nồng độ 3/1.000 ngâm giống 24-36 giờ sau đó vớt ra đãi sạch rồi đem ủ bình thường.\nSử dụng một trong các loại thuốc như: Kacie 250EC, Golcol 20SL, Supercin 20EC/40EC/80EC, Carbenzim 500FL, Tilt Super 300EC, Viroval 50BTN, Workup 9SL... để phun xịt. Về liều lượng và cách sử dụng nên đọc kỹ hướng dẫn của nhà sản xuất có in sẵn trên bao bì."
sauducthanlua = "Xử lý bệnh sâu đục thân lúa:\n Sử dụng sản phẩm TT Checker 270SC, liều lượng 40ml/bình 25L.\nBổ sung thêm cho cây lúa chất điều hòa sinh trưởng như Plastimula 1SL với liều lượng 30ml/bình 25L ở các thời kỳ quan trọng như: xử lý giống, đẻ nhánh, làm đòng."
suongmai = "Xử lý bệnh sương mai:\nSử dụng chế phẩm sinh học trừ bệnh RV04 với cơ chế tác động kép của nấm đối kháng và Enzyme kích kháng cây trồng tiêu diệt, cô lập các vết bệnh do nấm và vi khuẩn tấn công, ngăn chặn kịp thời sự lây lan của mầm bệnh.\nDựa vào điều kiện thời tiết mà nên tham khảo sử dụng để phòng bệnh từ đầu."
bogai = "Xử lý bọ gai\nLàm sạch cỏ dại trong ruộng và bờ bao.\nDiệt trừ sâu non trên mạ mùa sắp cấy bằng cách ngắt bỏ các lá bị hại có bọ gai.\nDùng thuốc trừ sâu nhóm lân hữu cơ, Carbamate hoặc Cúc tổng hợp đều có thể diệt được bọ gai trưởng thành và ấu trùng."
tungro = "Xử lý virus Tungro:\nPhun các loại thuốc trừ sâu có gốc buprofezin hay pymetrozine ở thời điểm ngày thứ 15 và ngày thứ 30 sau khi cấy có thể đạt hiệu quả nếu được thực hiện đúng thời gian. Các loài cây quanh cánh đồng cũng cần được phun các loại thuốc trừ sâu nêu trên.\nKhông nên sử dụng các sản phẩm thuốc trừ sâu có gốc chlorpyriphos, lamda cyhalothrin hay các công thức kết hợp các chất pyrethroid tổng hợp vì các loài sâu rầy đã phần nào đề kháng được các loại thuốc ấy. "
binhthuong = "Lúa không bị bệnh hoặc không bị bệnh do virus / vi khuẩn / nấm"

# Params
params = {
    'model': 'efficientnet_b3',
    'pretrained': False,
    'inp_channels': 3,
    'im_size': 300,
    'device': device,
    'batch_size': 85,
    'num_workers' : 0,
    'out_features': 11,
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
st.write('[Lưu ý:]')
luu_y = [
    'Sản phẩm chỉ sử dụng được cho lúa, xin vui lòng không nhập ảnh của các vật khác vào app',
    'Nhà phát triển không đảm bảo độ chính xác của dự đoán cho tất cả các dự đoán dựa trên ảnh của các giống cây ngoài lúa',
    'Chỉ có tác dụng với ảnh chụp từ điện thoại (tỉ lệ 4:3, độ phân giải khuyến khích: 1440x1080)'
]
st.write(luu_y)
        
st.sidebar.header('Tải ảnh lên để bắt đầu chẩn đoán')
uploaded_files = st.sidebar.file_uploader("Tải ảnh", accept_multiple_files=True)

res_col, img_col = st.columns([4, 2])

if uploaded_files is not None:
    try: 
        if os.path.exists('./upload'):
            shutil.rmtree('./upload')
        os.mkdir('./upload')
        upload_path = "./upload"

        # os.chdir('./database')
        # for uploaded_file in uploaded_files:
        #     img = load_image(uploaded_file)
        #     with open(uploaded_file.name, "wb") as f:
        #         f.write(uploaded_file.getbuffer())
        # os.chdir('../')

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
                    predictions = model(images).softmax(dim=1).argmax(dim=1).to('cpu').numpy()
                    
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
                    st.write(bacla)
                elif i == 1:
                    st.write(domsocla)
                elif i == 2:
                    st.write(bacchuyla)
                elif i == 3:
                    st.write(chaylua)
                elif i == 4:
                    st.write(domnau)
                elif i == 5:
                    st.write(sauducthanlua)
                elif i == 6:
                    st.write(suongmai)
                elif i == 7:
                    st.write(bogai)
                elif i == 8:
                    st.write(binhthuong)
                elif i == 9:
                    st.write(tungro)  
                elif i == 10:
                    st.write('Ảnh không xác định (Không phải ảnh lúa / chất lượng ảnh không phù hợp)')   

        with img_col:
            for img_path in os.listdir(upload_path)[:2]:
                image = Image.open(os.path.join('./upload', img_path))
                st.image(image, use_column_width='auto')

    except:
        pass

        




