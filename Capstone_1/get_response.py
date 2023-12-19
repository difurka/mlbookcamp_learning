

import requests
import os
import json
from skimage.io import imread
from skimage.transform import resize

url = 'http://localhost:9696/predict'

def get_test_img():
        
    test_dir = './test_model'
    for root, dirs, files in os.walk(os.path.join(test_dir)):
        if root.endswith('_Dermoscopic_Image'):
            x = (imread(os.path.join(root, files[0])))
        if root.endswith('_lesion'):
            y = (imread(os.path.join(root, files[0])))

    size = (256, 256)
    X = resize(x, size, mode='constant', anti_aliasing=True,)
    Y = resize(y, size, mode='constant', anti_aliasing=False) > 0.5

    

    return X.tolist(), Y.tolist()



image, lesion = get_test_img()
test_img = {
     'image': json.dumps(image),
     'lesion':json.dumps(lesion)
}


response = requests.post(url, json=test_img).json()
print('mertic for this image: ', response)

