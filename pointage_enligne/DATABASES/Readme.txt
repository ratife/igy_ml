
TUTORIEL DE CREATION DE LA BASE DE DONNEES ET SES TABLE

base.sql contient les requêtes simple de créaction de la base de données sans spécifié les port and host.
table.sql contient les requêtes de création des tables employee_base and mouvement dans une base de données.
psql -d db_name -a -f file.sql user: pour installer les tables dans une base de données spécifiques
exemple: psql -d mouvement_employee -a -f table.sql postgres

psql -U username dbname < mouvement_employee.pgsql: To IMPORT THIS DATABASE;

