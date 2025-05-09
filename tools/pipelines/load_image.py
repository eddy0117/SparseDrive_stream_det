import mmcv

def load_image(path,
               result_dict: dict):
    
    img_name = path + result_dict['sd_rec_c']['filename']
    img_ori = mmcv.imread(img_name, 'unchanged')
    return img_ori