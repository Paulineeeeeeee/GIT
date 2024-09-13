# for main model
python -m generativeimage2text.train -p "{'type': 'mytrain', \
'param':{'num_image_with_embedding':8}, \
'args' :{ \
      'num_workers':4, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':9,    \
      'bs':32 ,     \
      'acc_step':8, \
      'pat':2,      \
      'ckpt_path':'/home/pauline/checkpoint/', \
      'exp_name' :'LLAMATOUCH'\
      }}" 
      # 'load_path':'/data/cv/poyang/checkpoint/final_2lr_8img_ep1_lr1e-05_wd1e-05_im8.ckpt',\
      # 'exp_name' :'1000_2lr_8img_lowWD' \
python -m generativeimage2text.streaming_train -p "{'type': 'mytrain', \
'param':{'num_image_with_embedding':8}, \
'args' :{ \
      'num_workers':4, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':9,    \
      'bs':32 ,     \
      'acc_step':8, \
      'pat':2,      \
      'ckpt_path':'/home/pauline/checkpoint/', \
      'exp_name' :'LLAMATOUCH'\
      }}" 

python -m generativeimage2text.train -p "{'type': 'mytrain', \
'param':{'num_image_with_embedding':8}, \
'args' :{ \
      'num_workers':4, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':9,    \
      'bs':32 ,     \
      'acc_step':8, \
      'pat':2,      \
      'ckpt_path':'/home/pauline/checkpoint/', \
      'exp_name' :'AITW'\
      }}" 

# for pretrain model
python -m pretrain_llava.train -p "{'type': 'mytrain', \
'param':{'num_image_with_embedding':8}, \
'args' :{ \
      'num_workers':4, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':9,    \
      'bs':32 ,     \
      'acc_step':8, \
      'pat':2,      \
      'ckpt_path':'/home/pauline/checkpoint/', \
      'exp_name' :'pretrain_llava'\
      }}" 