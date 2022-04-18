import imp
from typing import Tuple, List
import sys
import cv2
import editdistance
from path import Path

from data_handle.data_preprocessor import DataPreprocess
from data_handle.data_load_lmdb import DataLoadLmdb,Batch
from model.rcnn_ctc_model import RCNNCTCModel

from collections import OrderedDict
from util.visualizer import Visualizer
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np

class Option:
    def __init__(self) -> None:
       self.display_id = 1
       self.display_freq = 400
       self.isTrain = True
       self.no_html = True
       self.display_winsize = 256
       self.name = 'user_log'
       self.display_port = 8192
       self.display_ncols = 4
       self.display_server = 'http://localhost'
       self.display_env = 'main'
       self.checkpoints_dir = ''

def train(model: RCNNCTCModel, loader: DataLoadLmdb, line_mode, learning_stagnation: int = 10) -> None:
    epoch = 0
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = DataPreprocess((256, 32) if line_mode else (128, 32), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  
    no_improvement_since = 0

    visualizer = Visualizer(Option()) 
    visualizer.reset()
    char_error_rate = 1
    word_accuracy = 0
    while True:
        epoch += 1
        print('--------------------Begin Train--------------------')
        loader.train_set()
        errors_ret = OrderedDict()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(line_mode, batch)
            loss = model.train_batch(batch)
            #print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
            
            # show visdom
            errors_ret['loss'] = loss
            errors_ret['Character_Error_Rate'] = char_error_rate*100
            errors_ret['Word_Accuracy'] = word_accuracy*100
            visualizer.plot_current_losses(epoch, float(iter_info[0]/iter_info[1]), errors_ret)
            visualizer.save_train_log(epoch, iter_info[0], loss)
        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)  

        # save mode or not
        if char_error_rate < best_char_error_rate:
            print('--------------------Error rate improved !!!--------------------')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
            print('Model saved')
        else:
            print('--------------------Error rate not improved !!!--------------------')
            print(f'Best rate: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        if no_improvement_since >= learning_stagnation:
            print(f'No more improvement since {learning_stagnation} epochs. Training stopped.')
            break
    print(f'--------------------ALL {epoch} Epoch Summary --------------------')
    message = f'Summary_char_error_rates: {summary_char_error_rates}  Summary_word_accuracies: {summary_word_accuracies}'
    print(message)
    visualizer.save_log(message)
    print(f'--------------------ALL {epoch} Epoch Summary END --------------------')


def validate(model: RCNNCTCModel, loader: DataLoadLmdb, line_mode: bool) -> Tuple[float, float]:
    print('--------------------Begin Validate--------------------')
    loader.validation_set()
    preprocessor = DataPreprocess((256, 32) if line_mode else (128, 32), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    visualizer = Visualizer(Option()) 
    
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        #print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(line_mode,batch)
        recognized, _ = model.infer_batch(batch)

        visualizer.reset()
        recognize_results = []
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            #print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->','"' + recognized[i] + '"')
            
            # show visdom
            img =  plt.imread(batch.file_names[i])
            if np.any(img):
                visualizer.display_img(img)

            recognize_result = {'real':batch.gt_texts[i],'recognized':recognized[i]}
            recognize_results.append(recognize_result)
            visualizer.display_result(recognize_results)
            visualizer.save_validate_log(iter_info[1],iter_info[0],batch.gt_texts[i],recognized[i],dist)
            
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print('--------------------Validate Summary --------------------')
    message = f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.'
    print(message)
    visualizer.save_log(message)
    print('--------------------Validate Summary END --------------------')
    return char_error_rate, word_accuracy


def infer(model: RCNNCTCModel, fn_img: Path) -> None:
    print('--------------------Begin Infer--------------------')
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = DataPreprocess((128, 32), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1, fn_img)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')

def main():
    command = sys.argv[1]
    batch_size = 500
    datasets_split = 0.95
    lmdb_path = Path("datasets_IAM")
    line_mode = False
    infer_img_path = "datasets_IAM/samples_wordImages/a01-000u/a01-000u-03-05.png"

    if command == 'train':
        loader = DataLoadLmdb(lmdb_path, batch_size, datasets_split)
        model = RCNNCTCModel(loader.char_list, 0)
        train(model, loader, line_mode)

    elif command == 'infer':
        with open("datasets_IAM/charList.txt") as f:
            fn_char_list = list(f.read())
        model = RCNNCTCModel(fn_char_list, must_restore=True)
        infer(model, infer_img_path)


if __name__ == '__main__':
    main()
