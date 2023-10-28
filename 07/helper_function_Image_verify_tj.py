#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)


# In[2]:


base_dir = r'/mnt/d/home/tjamil/MyWork/Hyper_Colab/NewWaveDigital/'   # Projet root dir


# In[3]:


train_data_dir = base_dir + r'Model_Create/data/training_images/'
test_data_dir = base_dir + r'Model_Evaluate/data/test_images/'
eval_data_dir = base_dir + r'Model_Evaluate/data/eval_images/'


# In[4]:


import os
import pathlib
from pathlib import Path
import imghdr
from PIL import Image
from datetime import datetime
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


# In[5]:


def image_verify(fpath):
    valid_ext = ['jpeg', 'png', 'gif', 'bmp', 'jpg'] #list of valid files ext.
    
    try:
        ext = pathlib.Path(fpath).suffix.lower()[1:]
        if ext not in valid_ext:
            return None
        else:
            image = Image.open(fpath)
            image.load()
            image.transpose(Image.FLIP_LEFT_RIGHT)
            image.close()
            img_type = imghdr.what(fpath)
            return (img_type)
    except:# (OSError, IOError, AttributeError) as e:
        return None
        #print('Bad image: %s' % e)
        #if image: image.close()
        #return e


# In[6]:


def helper_func(fp, app_list, msg):
        os.remove(fp)
        app_list.append(fp)
        print(msg, fp)

def files_cleansing(root_dir):
    
    total_files=0; process_dir = root_dir.split('/')[-2]
    list_ext=[]; tiny_files=[]; inval_images=[]
    
    msg = "Cleansing Files from '{}' folder (recursively).".format(process_dir)
    print(f"{msg}\n{'-'*len(msg)}", end='')
    
    for root, dirs, files in os.walk(root_dir):
        tot = len(files); fc=0; 
        if tot==0: continue  # no files but folder
        print('\n{0}; {1}'.format(total_files, tot), end=' ')
    
        for file in files:
            file_namef = os.path.join(root, file)
            file_ext   = pathlib.Path(file).suffix.lower()
            list_ext.append(file_ext)
            
            fc += 1; total_files += 1
            if fc % 20 == 0: print('.', end='')
    
            img_ext = image_verify(file_namef)
            if not(img_ext): #image/ext not verified
                helper_func(file_namef, inval_images ,'\nImage not verified:')
                continue

            if (os.stat(file_namef).st_size < 2000):
                helper_func(file_namef, tiny_files ,'\nTiny/low-res image:')
                continue
                 
    print('\nTotal files sacnned : {0}\n' 
            'Invlid-images       : {1}\n' 'Tiny/low-res images : {2}\n\n'
    .format(total_files, len(inval_images), len(tiny_files)))
    
    log_file_path = root_dir.rsplit('/',2)[0]
    
    dated = datetime.now().strftime("%Y_%m_%d-%H")
 
    log_file = f"{log_file_path}/files_removed_from_{process_dir}_{dated}.txt"
    with open(log_file, "+w") as outfile:
        outfile.write("\n".join(inval_images)),
        outfile.write("\n".join(tiny_files))
    print('Deleted files log saved in {}\n'.format(log_file))
    return (inval_images, tiny_files)


# In[7]:


def classes_to_ranks(scores):
    conf_values  = nn.softmax(scores)
    ranks = [sorted(conf_values).index(x) for x in conf_values]
    classes = [x.replace('_Images', '') for x in class_names]
    tupled_ranks = [(x, classes[ranks.index(x)], '{0:3.2f}%'.format(conf_values[ranks.index(x)]*100)) for x in ranks]
    return sorted(tupled_ranks, key = lambda x: x[0], reverse=True)


# ### *** END OF MODULE ***




