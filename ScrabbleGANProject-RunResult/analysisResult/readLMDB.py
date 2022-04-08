# todo: Extract data (or pictures) from generated LMDB format files

from base64 import encode
from PIL import Image
import lmdb
import caffe2

def read_from_lmdb(lmdb_path,img_save_to):
 try:
  lmdb_env=lmdb.open(lmdb_path, map_size=3221225472)
  lmdb_txn=lmdb_env.begin()
  lmdb_cursor=lmdb_txn.cursor()
  datum=caffe2.Datum()
 
  datum_index=0
  for key,value in lmdb_cursor:
    datum.ParseFromString(value)
    label=datum.label
    data=datum.data
    channel=datum.channels
    print ('Datum channels: %d' % datum.channels)
    print ('Datum width: %d' % datum.width)
    print ('Datum height: %d' % datum.height)
    print ('Datum data length: %d' % len(datum.data))
    print ('Datum label: %d' % datum.label)
 
    size=datum.width*datum.height
    pixles1=datum.data[0:size]
    pixles2=datum.data[size:2*size]
    pixles3=datum.data[2*size:3*size]
    #Extract images of different channel
    image1=Image.frombytes('L', (datum.width, datum.height), pixles1)
    image2=Image.frombytes('L', (datum.width, datum.height), pixles2)
    image3=Image.frombytes('L', (datum.width, datum.height), pixles3)
    
    image4=Image.merge("RGB",(image3,image2,image1))
    image4.save(img_save_to+str(key)+".jpg")
    datum_index+=1
    print ("extracted")
 
 finally:
   lmdb_env.close()

def read_from_lmdb2(lmdb_path):
    env_db = lmdb.open(lmdb_path) 
    txn = env_db.begin() 
    print(txn.get(("image-000000001").encode()))
    print()
    for key, value in txn.cursor():
        print(key, value) 
        print(123)
    env_db.close()


read_from_lmdb2("E:\data")