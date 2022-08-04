import sys


class Progressbar:
    def update(self, percent):
        if not self.file.isatty():
            return
        width = 60
        x = int(percent/100.0*width)
        if self.need_return:
            self.file.write('\r')
        self.file.write(f'[{"#"*x}{"."*(width-x)}] {percent:.2f} %')
        self.file.flush()
        self.need_return = True

    def remove(self):
        if not self.file.isatty():
            return
        if self.need_return:
            self.file.write('\r')
        self.file.write(f'{" "*self.total_width}\r')
        self.file.flush()

    def __init__(self, width=60):
        self.width = width
        self.total_width = width+11
        self.file = sys.stdout
        self.need_return = False

    def __del__(self):
        self.remove()
