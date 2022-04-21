import configparser

from PIL import Image, ImageOps

# FEI tags containing image details
TIFFTAG_META_PHENOM = 34683
TIFFTAG_META_QUANTA = 34682
TIFFTAGS = [TIFFTAG_META_PHENOM, TIFFTAG_META_QUANTA]

class SEMImage:
    def extract_tags(self):
        sem_metadata = None
        exif = self.img.getexif()
        for tag in exif:
            if tag in TIFFTAGS:
                sem_metadata = exif.get(tag)
                break
        if sem_metadata is None:
            return None
        
        config = configparser.ConfigParser()
        config.read_string(sem_metadata)

        if 'Scan' in config:
            self.scale_x = config['Scan'].get('PixelWidth')
            self.scale_y = config['Scan'].get('PixelHeight')

        if 'Stage' in config:
            self.rotation = config['Stage'].get('StageT')
            
        if 'PrivateFei' in config:
            databar_height = config['PrivateFei'].get('DatabarHeight','0')
            self.img = self.img.crop((0,0,self.img.width,self.img.height-int(databar_height)))

    def __init__(self, filename, scale=1):
        self.img = Image.open(filename)
        self.extract_tags()
        if scale != 1:
            self.img = ImageOps.scale(self.img, scale)
