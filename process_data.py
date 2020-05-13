from PIL import Image


def process_data(image, target_size=(448, 448)):
    if hasattr(image, '_getexif'):
        dict_exif = image._getexif()
        if dict_exif == None:
            new_img = image
        elif dict_exif.get(274) == 3:
            new_img = image.rotate(180)
        elif dict_exif.get(274) == 6:
            new_img = image.rotate(-90)
        else:
            new_img = image
    else:
        new_img = image
    iw, ih = new_img.size
    w, h = target_size
    rate = min(w / iw, h / ih)
    nw = int(iw * rate)
    nh = int(ih * rate)
    resize_image = image.resize((nw, nh), Image.BICUBIC)
    image_paste = Image.new('RGB', target_size, (128, 128, 128))
    image_paste.paste(resize_image, ((w - nw) // 2, (h - nh) // 2))
    # image_paste.save('save.jpg')
    return image_paste
