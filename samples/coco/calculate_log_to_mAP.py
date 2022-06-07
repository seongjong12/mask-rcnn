import csv
import os
import numpy as np
from collections import defaultdict

def calculate_AP(cat_id):
    print('cat_id :', cat_id)
    num_image = 0
    sum_precision = 0
    
    file_pointer = open('log_' + str(cat_id) + '.csv','r')
    csv_reader = csv.reader(file_pointer)

    line_dict_image = defaultdict(list) 

    for line in csv_reader:
        # idx_line = int(line[0])
        image_id = line[3]
        confusion_value = line[9]
        line_dict_image[image_id].append(confusion_value)

    key_list = list(line_dict_image.keys())
    if len(key_list) == 0:
        return -1
    
    for idx_image, key in enumerate(key_list):
        confusion_values = line_dict_image[key]

        TP_num = 0
        FP_num = 0
        FN_num = 0
        
        for values in confusion_values:
            if values == 'FP':
                FP_num += 1
            elif values == 'FN':
                FN_num += 1
            elif values == 'TP':
                TP_num += 1

        precision = TP_num / (TP_num + FP_num + np.spacing(1))
        sum_precision += precision
        num_image += 1
        
    Average_precision = sum_precision / (idx_image + 1)
    return Average_precision

mAP = 0
count_mi = 0
for cat_id in range(1, 15):
    Average_precision = calculate_AP(cat_id)
    if not Average_precision == -1:
        mAP += Average_precision
    else: 
        count_mi += 1

print('mAP :', mAP / (13-count_mi))



# def calculate_mAP():
#     num_image = 0
#     sum_precision = 0

#     for cat_id in range(1, 14):
#         print(cat_id)
#         file_pointer = open('log_' + str(cat_id) + '.csv','r')
#         csv_reader = csv.reader(file_pointer)

#         line_dict_image = defaultdict(list) 

#         for line in csv_reader:
#             # idx_line = int(line[0])
#             image_id = line[3]
#             confusion_value = line[9]
#             line_dict_image[image_id].append(confusion_value)

#         key_list = list(line_dict_image.keys())
#         if len(key_list) == 0:
#             continue
        
#         for idx_image, key in enumerate(key_list):
#             confusion_values = line_dict_image[key]

#             TP_num = 0
#             FP_num = 0
#             FN_num = 0
            
#             for values in confusion_values:
#                 if values == 'FP':
#                     FP_num += 1
#                 elif values == 'FN':
#                     FN_num += 1
#                 elif values == 'TP':
#                     TP_num += 1

#             precision = TP_num / (TP_num + FP_num + np.spacing(1))
#             sum_precision += precision
#             num_image += 1
        
#     Average_precision = sum_precision / (idx_image + 1)
#     return Average_precision