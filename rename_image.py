import os
os.getcwd()
collection = "./img/images_Q/"

for i, filename in enumerate(os.listdir(collection)):
    os.rename("./img/images_Q/" + filename, "./img/images_Q/" + "image_" + str(i + 1) + ".jpg")