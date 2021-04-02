import sqlite3

sqliteConnection = sqlite3.connect('threads.db')
cursor = sqliteConnection.cursor()
print("Connected to SQLite")
#sqliteConnection.execute("""CREATE TABLE IF NOT EXISTS cones(id INTEGER PRIMARY KEY, roi BINARY UNIQUE)""")
max_id = cursor.execute("""SELECT count(id) FROM cones""").fetchall()[0][0]
print(max_id)
