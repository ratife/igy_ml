import xmlrpc.client
from time import gmtime, strftime,time
#from odoo import models, fields, api, exceptions




def WebServ(HOST,PORT,DB,USER,PASS,ID_EMP):
	q= time()
	ROOT = 'http://%s:%d' % (HOST,PORT) 
	try:
		common = xmlrpc.client.ServerProxy(ROOT+'/xmlrpc/2/common')
		models = xmlrpc.client.ServerProxy(ROOT+'/xmlrpc/2/object')
		uid = common.authenticate(DB,USER, PASS,{})
		if uid == False:
			print('Veuillez verifier les parametres HOST,PORT,DB,USER')
		# coordonne liste personnel
		personnel = {'employee_id': ID_EMP,'check_in': strftime("%Y-%m-%d %H:%M:%S")}
		#checkin = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'search_read', [[['employee_id','=',1],['check_in','=',False]]])
		#if len(checkin) !=0:
		print('zezeezezezez')
		try:
			new_contact_id = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'create', [personnel])
			print('presonnel add checkin ...')
			# check = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'search_read', [[['employee_id','=',1],['check_out','=',False]]])
			# print(check)
			# if len(check ) == 1:
			# 	check_out_odoo = {'check_out': '2021-04-29 17:29:49'}
			# 	# check_out_odoo = {'check_out': strftime("%Y-%m-%d %H:%M:%S")}
			# 	models.execute_kw(DB, uid, PASS, 'hr.attendance', 'write', [check[0], check_out_odoo])
		except :
			#print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
			#check = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'search_read', [[['employee_id','=',ID_EMP],['check_out','=',False]]])
			check1 = models.execute_kw(DB, uid, PASS, 'hr.attendance', 'search', [[['employee_id','=',ID_EMP],['check_out','=',False]]])
			#print("check1:",check1)
			#print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
			#print("chexk...",check)
			if len(check1 ) == 1:
				#check_out_odoo = {'check_out': strftime("%Y-%m-%d %H:%M:%S")}
				models.execute_kw(DB, uid, PASS, 'hr.attendance', 'write', [check1, {'check_out': strftime("%Y-%m-%d %H:%M:%S")}])
			print('checkout done ...')
	except ConnectionRefusedError:
		print('ERREUR DE CONNEXION')
	r = time()
	print('temps d execution', r-q)

HOST = 'localhost'
PORT = 8069
DB = 'avotra_data'
USER = 'radez0509@gmail.com'
PASS = '19091994'
ROOT = 'http://%s:%d' % (HOST,PORT)
ID_EMP = 5

WebServ(HOST,PORT,DB,USER,PASS,ID_EMP)
