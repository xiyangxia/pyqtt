from PyQt5.QtWidgets import QMessageBox


class Message(QMessageBox):
    timeout = 0
    autoClose = True
    currentTime = 0
    def showEvent(self, QShowEvent):
        currentTime = 0
        if self.autoClose:
            self.startTimer(1000)

    def timerEvent(self, *args, **kwargs):
        self.currentTime+=1
        if (self.currentTime >= self.timeout):
            self.done(0)




