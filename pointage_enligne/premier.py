import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtg
class MainWindow(qt.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Izy mitsam!")

        #Layout
        self.setLayout(qt.QVBoxLayout())

        my_label = qt.QLabel("Pointage!")
        my_label.setFont(qtg.QFont('Helvetice', 18))
        self.layout().addWidget(my_label)
        # entry 
        my_entry = qt.QLineEdit()
        my_entry.setObjectName("Name_Field: ")
        my_entry.setText("")
        self.layout().addWidget(my_entry)

        #Button
        my_button = qt.QPushButton("Press", clicked = lambda: press_it())
        self.layout().addWidget(my_button)
        self.show()

        def press_it():
            my_label.setText(f' Hello{my_entry.text}')
            my_entry.setText(" ")



app = qt.QApplication([])

mw = MainWindow()
app.exec_()
