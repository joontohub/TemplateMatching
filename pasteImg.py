from PIL import Image

resized = Image.open('resized.jpg')
original = Image.open('original.jpg')

new_image = original
new_image.paste(resized, (startX, startY, endX, endY))
new_image.save("paste_result.jpg")
