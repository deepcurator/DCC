import psycopg2
from pathlib import Path
from configparser import ConfigParser


class Database:
    cred_path = Path("../config/credentials.cfg").resolve()
    config = ConfigParser()
    config.read(str(cred_path))

    def __init__(self):
        self.connection = psycopg2.connect(user=self.config.get("Database", "user"),
                                           password=self.config.get(
                                               "Database", "password"),
                                           host=self.config.get(
                                               "Database", "host"),
                                           port=self.config.get(
                                               "Database", "port"),
                                           database=self.config.get("Database", "database"))
        self.cursor = self.connection.cursor()

    def upsert_query(self, values_list: list):
        """Inserts the values into table if they do not exist."""
        # values should be a list of tuple/list containing values for a row
        try:
            values = ",".join(
                ("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)") % tup for tup in values_list)
        except:
            raise ValueError(
                "The values_list do not have proper number of values!")

        column_list = ("(dir_name, title, framework, lightweight, err_msg,"
                       "pdate, tags, stars, code_link, paper_link)")
        query = ("INSERT INTO metadata %s"
                 "VALUES %s"
                 "ON CONFLICT (dir_name) DO NOTHING;")
        self.cursor.execute(query, (column_list, values))
        self.connection.commit()

    def update_query(self, dir_name: str, status: str, err_msg: str):
        query = ("UPDATE metadata"
                 "SET lightweight = %s,"
                 "err_msg = %s"
                 "WHERE dir_name = %s;")
        self.cursor.execute(query, (status, err_msg, dir_name))
        self.connection.commit()

    def __del__(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()
            print("PostgreSQL connection is closed")
