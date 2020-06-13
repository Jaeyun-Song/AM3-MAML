import h5py
import os
import shutil
import io
from PIL import Image
import numpy as np
from tqdm import tqdm

def main(path):
    # open h5py files
    datapath = os.path.join('../data','cub',path)
    new_datapath = os.path.join('../data','cub','new_'+path)
    shutil.copy(datapath, new_datapath)
    data = h5py.File(new_datapath, 'r+')['datasets']
    for k in tqdm(data.keys()):
        for i, img in enumerate(data[k]):
            img = Image.open(io.BytesIO(img)).convert('RGB')
            img = img.resize([84,84])
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format='PNG')
            imgByteArr = np.frombuffer(imgByteArr.getvalue(), np.uint8)
            data[k][i] = imgByteArr
            new_img = Image.open(io.BytesIO(data[k][i])).convert('RGB')

if __name__=='__main__':
    print('train')
    main('train_data.hdf5')
    print('valid')
    main('val_data.hdf5')
    print('test')
    main('test_data.hdf5')