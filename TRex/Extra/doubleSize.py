from PIL import Image

img=Image.open("sprites/touched/low1.png")
size=list(img.size)
size[0] = 118
size[1] = 66
downsized=img.resize(size, Image.NEAREST) # NEAREST drops the lines
downsized.save("sprites/touched/low1.png")

img=Image.open("sprites/touched/low2.png")
size=list(img.size)
size[0] = 118
size[1] = 66
downsized=img.resize(size, Image.NEAREST) # NEAREST drops the lines
downsized.save("sprites/touched/low2.png")
