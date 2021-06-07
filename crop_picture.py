from PIL import Image

file_name = "synthesis.png"
img = Image.open(file_name)
area = (15, 15, 235, 235)
cropped_img = img.crop(area)

img.show()


cropped_img.show()


cropped_img.save("./cropped_" + file_name)
