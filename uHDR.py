# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr




"""Only contains main program.
"""


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDesktopWidget
import guiQt.controller
from multiprocessing import freeze_support

import sys
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    freeze_support()
    print("uHDRv6 (C++ core)")

    app = QApplication(sys.argv)

    mcQt = guiQt.controller.AppController(app)

    sys.exit(app.exec_())
# ------------------------------------------------------------------------------------------
