
TUTORIEL DE CREATION DE LA BASE DE DONNEES ET SES TABLE

base.sql contient les requ�tes simple de cr�action de la base de donn�es sans sp�cifi� les port and host.
table.sql contient les requ�tes de cr�ation des tables employee_base and mouvement dans une base de donn�es.
psql -d db_name -a -f file.sql user: pour installer les tables dans une base de donn�es sp�cifiques
exemple: psql -d mouvement_employee -a -f table.sql postgres

psql -U username dbname < mouvement_employee.pgsql: To IMPORT THIS DATABASE;

