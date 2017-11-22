import random
import requests


URL = 'https://cdn.gratisography.com/photos/{0}H.jpg'
MAX_VAL = 447

def get_image():
    random_img_num = random.randint(1, MAX_VAL+1)
    img_url = URL.format(random_img_num)
    img_path = 'input/gratisography_{0}H.jpg'.format(random_img_num)
    resp = requests.get(img_url, stream=True)
    if resp.status_code == 200:
        with open(img_path, 'wb') as f:
            for chunk in resp:
                f.write(chunk)
    return img_path


print(get_image())
