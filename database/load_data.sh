#!/bin/bash

authors_db_file="authors.sql"
postgres_db_file="postgres.sql"

sudo -u postgres psql -c "DROP table taggedarticles"
sudo -u postgres psql -c "DROP DATABASE authors"
sudo -u postgres psql -c "CREATE DATABASE authors"

sudo -u postgres psql postgres -f "$postgres_db_file"
sudo -u postgres psql authors -f "$authors_db_file"
