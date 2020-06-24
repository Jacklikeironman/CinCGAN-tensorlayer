from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

##G1,G2,D1 Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 2 * (1e-4)
config.TRAIN.beta1 = 0.5
config.TRAIN.n_epoch_init = 2000
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = int(config.TRAIN.n_epoch_init / 10)

config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_BICUBIC_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'
config.TRAIN.lr_UNKNOWN_img_path = 'data2017/DIV2K_train_LR_unknown/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_BICUBIC_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'
config.VALID.lr_UNKNOWN_img_path = 'data2017/DIV2K_valid_LR_unknown/X4/'

