from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, QMessageBox, QVBoxLayout
import sys
import MySQLdb as mdb
 
 
class Window(QDialog):
    def __init__(self):
        super().__init__()
 
        self.title = "PyQt5 Insert Data To Employee"
        self.top = 600
        self.left = 600
        self.width = 600
        self.height = 600
 
 
        self.InitWindow()
 
 
    def InitWindow(self):
 
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
 
        vbox = QVBoxLayout()
 
        self.nom = QLineEdit(self)
        self.nom.setPlaceholderText('Entrer nom: ')
        self.nom.setStyleSheet('background:cyan')
        self.nom.setFont(QtGui.QFont("Sanserif", 15))
        
        vbox.addWidget(self.nom)
 
        self.prenom = QLineEdit(self)
        self.prenom.setPlaceholderText('Entrer prenom:')
        self.prenom.setStyleSheet('background:cyan')
        self.prenom.setFont(QtGui.QFont("Sanserif", 15))
 
        vbox.addWidget(self.prenom)
 
        self.id_odoo = QLineEdit(self)
        self.id_odoo.setPlaceholderText('Entrer l\'id_odoo')
        self.id_odoo.setFont(QtGui.QFont("Sanserif", 15))
        self.id_odoo.setStyleSheet('background:cyan')
 
        vbox.addWidget(self.id_odoo)

        self.date_creation = QLineEdit(self)
        self.date_creation.setPlaceholderText('Date:')
        self.date_creation.setFont(QtGui.QFont("Sanserif", 15))
        self.date_creation.setStyleSheet('background:cyan')
 
        vbox.addWidget(self.date_creation)
 
        self.button = QPushButton("Insert Data", self)
        self.button.setStyleSheet('background:magenta')
        self.button.setFont(QtGui.QFont("Sanserif", 15))
        self.button.clicked.connect(self.InsertData)
        
        vbox.addWidget(self.button)

        
        self.button = QPushButton("ShowData", self)
        self.button.setStyleSheet('background:magenta')
        self.button.setFont(QtGui.QFont("Sanserif", 15))
        self.button.clicked.connect(self.ShowData)
        
        vbox.addWidget(self.button)
 
       
        self.setLayout(vbox)
        self.show()
 
 
    def InsertData(self):
        con = mdb.connect('localhost', 'root', '', 'mouvement_employee')
        with con:
            cur = con.cursor()
 
            cur.execute("INSERT INTO employee_based(nom, prenom, id_odoo, date_creation)"
                        "VALUES('%s', '%s', '%s', '%s')" % (''.join(self.nom.text()),
                                                  ''.join(self.prenom.text()),''.join(self.id_odoo.text()),''.join(self.date_creation.text())))
 
 
            QMessageBox.about(self,'Connection', 'Data Inserted Successfully')
            self.close()
 
    def ShowData(self):
        con = mdb.connect('localhost', 'root', '', 'mouvement_employee')
        with con:
            cur = con.cursor()
 
            cur.execute("SELECT * FROM employee_based(nom, prenom, id_odoo, date_creation)")
 
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())