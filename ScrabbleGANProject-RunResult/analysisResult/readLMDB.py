# Extract data (or pictures) from generated LMDB format files
from base64 import encode
from PIL import Image
import lmdb
import os

def read_from_lmdb(lmdb_path):
    env_db = lmdb.open(lmdb_path, readonly=True) 
    with env_db.begin() as txn:
      for i in range(1000):
        #i = i + 115000
        img_key = 'image-%09d' % i
        print(img_key)
        data = txn.get(img_key.encode("utf-8"))
        if(data != None):
          img_name = str(i)+'.jpg'
          with open(os.path.join("../result2/"+img_name),'wb') as f:
              f.write(data)

read_from_lmdb("E:/Advaced Intro to Machine Learning _CSCI642/final project/convolutional-handwriting-gan-master/lmdb_files/finalProjectDemo2_IAM_concat100k")

