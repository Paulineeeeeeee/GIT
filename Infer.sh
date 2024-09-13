python -m generativeimage2text.train -p "{'type': 'myinfer', \
'param':{'num_image_with_embedding':8}, \
'args' :{ \
      'num_workers':4, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':1,    \
      'bs':32 ,     \
      'acc_step':8, \
      'pat':2,      \
      'load_path':'/home/pauline/checkpoint/test_ep1_lr1e-05_wd0.0001_im8.ckpt',\
      'ckpt_path':'/home/pauline/checkpoint/', \
      'exp_name' :'test'\
      }}" 
      # 'exp_name' :'1000_2lr_8img_lowWD' \