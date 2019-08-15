import psycopg2
from psycopg2 import sql
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

    def make_one_values_tuple(self, values: list):
        return sql.SQL("({})").format(sql.SQL(", ").join(sql.Literal(value) for value in values))

    def upsert_query(self, values_list: list):
        """Inserts the values into table if they do not exist."""
        # values should be a list of tuple/list containing values for a row

        values = sql.SQL(', ').join(
            (self.make_one_values_tuple(value) for value in values_list))

        query = sql.SQL("INSERT INTO metadata (dir_name, title, framework, lightweight, err_msg, pdate, tags, stars, code_link, paper_link)"
                        "VALUES {} "
                        "ON CONFLICT (dir_name) DO NOTHING;").format(values)
        self.cursor.execute(query)
        self.connection.commit()

    def update_query(self, dir_name: str, status: str, err_msg: str):
        query = sql.SQL("UPDATE metadata "
                        "SET lightweight = %s, "
                        "err_msg = %s "
                        "WHERE dir_name = %s;")
        self.cursor.execute(query, (status, err_msg, dir_name))
        self.connection.commit()

    def get_table(self) -> list:
        query = sql.SQL("SELECT * FROM metadata;")
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def __del__(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()
            print("PostgreSQL connection is closed")


if __name__ == "__main__":
    db = Database()
    table = db.get_table()
    pass
