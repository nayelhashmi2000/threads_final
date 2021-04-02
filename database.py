import sqlite3
import glob
import os

""" create a database connection to a SQLite database """
conn = None
try:
    conn = sqlite3.connect(r"threads.db")
    print(sqlite3.version)
except Error as e:
    print(e)
finally:   
#create_connection(r"threads.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS cones(id INTEGER PRIMARY KEY, roi BINARY UNIQUE)""")
    if conn:
        conn.close()

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData

def insertBLOB(Id, photo):
    try:
        sqliteConnection = sqlite3.connect('threads.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        #sqliteConnection.execute("""CREATE TABLE IF NOT EXISTS cones(id INTEGER PRIMARY KEY, roi BINARY UNIQUE)""")
        sqlite_insert_blob_query = """ INSERT INTO cones
                                  (id, roi) VALUES (?, ?)"""

        roi = convertToBinaryData(photo)
        # Convert data into tuple format
        data_tuple = (Id, roi)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
i = 1
for image in glob.glob("imgs_cropped\\*.jpg"):
    insertBLOB(i, image)
    i += 1
