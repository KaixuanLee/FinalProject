import cv2
import lmdb
import pickle
from path import Path

# lmdb path
lmdb_path = Path("../datasets_IAM")
# 2GB lmdb file
env = lmdb.open(str(lmdb_path / 'lmdb'), map_size=1024 * 1024 * 1024 * 2)
# go over all png files
fn_imgs = list((lmdb_path / 'wordImages').walkfiles('*.png'))
# put the imgs into lmdb
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        txn.put(basename.encode("ascii"), pickle.dumps(img))

env.close()