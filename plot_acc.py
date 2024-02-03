import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

save_dir = './files'

unet = pd.read_csv(save_dir + '/data.csv')

val_unet_epoch = unet[['epoch']]
val_dec_dice_coef = unet[['dice_coef']]
val_dec_val_dice_coef = unet[['val_dice_coef']]

plt.plot( val_unet_epoch[0:], val_dec_val_dice_coef[0:], label="Validation", linestyle="dashed", marker='o', color='orange', ms = 5,markevery=5)
plt.plot( val_unet_epoch[0:], val_dec_dice_coef[0:], label="Training",  marker='d',markevery=5, color='b', ms = 5)

plt.ylabel("Dice coefficient")
plt.xlabel("epochs")

plt.legend()
#plt.savefig('./results/stl/loss_dec_stl.png')
plt.show()


