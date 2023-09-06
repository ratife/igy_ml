import mysql.connector
import xmlrpc.client
from time import gmtime, strftime,time

def WebServ1(HOST,PORT,DB,USER,PASS,ID_EMP,date):
	ROOT = 'http://%s:%d' % (HOST,PORT) 
	try:
		common = xmlrpc.client.ServerProxy(ROOT+'/xmlrpc/2/common')
		models = xmlrpc.client.ServerProxy(ROOT+'/xmlrpc/2/object')
		uid = common.authenticate(DB,USER, PASS,{})
		if uid == False:
			print('Veuillez verifier les parametres HOST,PORT,DB,USER')
		personnel = {'employee_id': ID_EMP,'check_in': date}
		try:
			new_contact_id = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'create', [personnel])
			print('presonnel add checkin ...')
		except :
			check1 = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'search', [[['employee_id','=',ID_EMP],['check_out','=',False]]])
			if len(check1 ) == 1:
				models.execute_kw(DB, uid, PASS, 'hr.attendance', 'write', [check1, {'check_out': date}])
			print('checkout done ...')
	except ConnectionRefusedError:
		print('ERREUR DE CONNEXION')
	

def check_odoo ():
	mydb = mysql.connector.connect(host="localhost",user="root",password="",database="avotrah")
	mycursor = mydb.cursor()
	mycursor.execute('select id_odoo,date_mouvement from  employee_based inner join mouvement on employee_based.id = mouvement.employee_id where is_synchronise = 0')
	return mycursor.fetchall()

def database_update(val):
    mydb = mysql.connector.connect(host="localhost",user="root",password="",database="avotrah")
    mycursor = mydb.cursor()
    mycursor.execute('update mouvement set is_synchronise = 1 where employee_id=' +str(val)+ ' and is_synchronise = 0')
    mydb.commit()

def write_odoo():
	for i in check_odoo():
	    WebServ1(HOST,PORT,DB,USER,PASS,int(i[0]),str(i[1]))
	    database_update(int(i[0]))


def w1():
	j=10
	for i in check_odoo():
	    q='2021-06-10 15:'+str(j)+':51'
	    WebServ1(HOST,PORT,DB,USER,PASS,int(i[0]),q)
	    j=j+1

if __name__ == '__main__':
	w1()