import psycopg2
import psycopg2.extras
import pandas as pd 
import time
import random

class PostgreSQLConnector:
    def __init__(self, db_name):
        self.db_name = db_name 
        self.user = "read_write"
        self.password = ""
        self.host = ""
        self.port = "5432"

    def start_connection(self):
        self.connection = psycopg2.connect(
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            host=self.host,  # Assuming the database is hosted locally
            port=self.port        # Port number for PostgreSQL (default is 5432)
        )
        self.cursor = self.connection.cursor()
        
    def disconnect(self):
        self.cursor.close()
        self.connection.close()
        return

    def read(self, query, return_df = True):
        try:
            self.start_connection()
            self.cursor.execute(query)
            db_col = [desc[0] for desc in self.cursor.description]
            db_result = self.cursor.fetchall()

        except psycopg2.Error as err:
            print("Error executing PostgreSQL READ commands:", err)

        finally:
            self.disconnect()
            
        if (return_df):
            db_df = pd.DataFrame(db_result, columns = db_col)
            return db_df
        else:
            return db_col + db_result
    
    def write(self, query, data_values):
        try:
            self.start_connection()
            psycopg2.extras.execute_values (
                self.cursor, query, data_values
            )  
            self.connection.commit()

        except psycopg2.Error as err:
            print("Error executing PostgreSQL WRITE commands:", err)

        finally:
            self.disconnect()

        return 

    def delete(self, query):
        try:
            self.start_connection()
            self.cursor.execute(query)
            self.connection.commit()

        except psycopg2.Error as err:
            print("Error executing PostgreSQL DELETE commands:", err)
        
        finally:
            self.disconnect()

    def test_read(self):
        PG = PostgreSQLConnector(self.db_name)
        start = time.time()
        df = PG.read("SELECT * from public.datacamp_courses;", return_df = False)
        end = time.time()
        print(f"elapsed time: {end-start} seconds")
        print(df)

if __name__ == "__main__":
    PG = PostgreSQLConnector("test")
    PG.test_read()



    