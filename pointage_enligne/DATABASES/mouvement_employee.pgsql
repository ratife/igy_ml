--
-- PostgreSQL database dump
--

-- Dumped from database version 9.5.21
-- Dumped by pg_dump version 9.5.21

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: employee_base; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.employee_base (
    employee_id bigint NOT NULL,
    employee_nom character varying(255) NOT NULL,
    employee_prenom character varying(255) NOT NULL,
    employee_id_odoo integer NOT NULL,
    employee_date_creation date DEFAULT ('now'::text)::date NOT NULL
);


ALTER TABLE public.employee_base OWNER TO postgres;

--
-- Name: employee_base_employee_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.employee_base_employee_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.employee_base_employee_id_seq OWNER TO postgres;

--
-- Name: employee_base_employee_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.employee_base_employee_id_seq OWNED BY public.employee_base.employee_id;


--
-- Name: mouvement; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mouvement (
    mouvement_id bigint NOT NULL,
    employee_id integer NOT NULL,
    mouvement_date date DEFAULT now(),
    mouvement_login boolean DEFAULT false NOT NULL,
    mouvement_logout boolean DEFAULT false NOT NULL,
    mouvement_synchronise_odoo boolean DEFAULT false NOT NULL
);


ALTER TABLE public.mouvement OWNER TO postgres;

--
-- Name: mouvement_mouvement_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.mouvement_mouvement_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.mouvement_mouvement_id_seq OWNER TO postgres;

--
-- Name: mouvement_mouvement_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.mouvement_mouvement_id_seq OWNED BY public.mouvement.mouvement_id;


--
-- Name: employee_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employee_base ALTER COLUMN employee_id SET DEFAULT nextval('public.employee_base_employee_id_seq'::regclass);


--
-- Name: mouvement_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mouvement ALTER COLUMN mouvement_id SET DEFAULT nextval('public.mouvement_mouvement_id_seq'::regclass);


--
-- Data for Name: employee_base; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.employee_base (employee_id, employee_nom, employee_prenom, employee_id_odoo, employee_date_creation) FROM stdin;
\.


--
-- Name: employee_base_employee_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.employee_base_employee_id_seq', 1, false);


--
-- Data for Name: mouvement; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.mouvement (mouvement_id, employee_id, mouvement_date, mouvement_login, mouvement_logout, mouvement_synchronise_odoo) FROM stdin;
\.


--
-- Name: mouvement_mouvement_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.mouvement_mouvement_id_seq', 1, false);


--
-- Name: employee_base_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employee_base
    ADD CONSTRAINT employee_base_pkey PRIMARY KEY (employee_id);


--
-- Name: mouvement_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mouvement
    ADD CONSTRAINT mouvement_pkey PRIMARY KEY (mouvement_id);


--
-- Name: fk_employee; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mouvement
    ADD CONSTRAINT fk_employee FOREIGN KEY (employee_id) REFERENCES public.employee_base(employee_id) ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

