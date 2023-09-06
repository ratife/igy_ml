DROP TABLE IF EXISTS empoyee_base;
DROP TABLE IF EXISTS mouvement;

CREATE TABLE employee_base(
   employee_id BIGSERIAL NOT NULL ,
   employee_nom VARCHAR(255) NOT NULL,
   employee_prenom VARCHAR(255) NOT NULL,
   employee_id_odoo INT NOT NULL,
   employee_date_creation DATE NOT NULL DEFAULT CURRENT_DATE,
   PRIMARY KEY(employee_id)
);

CREATE TABLE mouvement(
   mouvement_id BIGSERIAL NOT NULL,
   employee_id INT NOT NULL,
   mouvement_date DATE DEFAULT Now(),
   mouvement_login BOOLEAN NOT NULL DEFAULT FALSE,
   mouvement_logout BOOLEAN NOT NULL DEFAULT FALSE,
   mouvement_synchronise_odoo BOOLEAN NOT NULL DEFAULT FALSE,
   
   PRIMARY KEY(mouvement_id),
   CONSTRAINT fk_employee
      FOREIGN KEY(employee_id) 
	  REFERENCES employee_base(employee_id)
	  ON DELETE CASCADE
);