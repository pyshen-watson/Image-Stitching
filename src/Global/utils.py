def isImage(str):
    exts = ['.jpg','.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']
    for ext in exts:
        if ext in str:
            return True
    return False
