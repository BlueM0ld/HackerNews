import pandas as pd
import psycopg2

conn_params = {
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao",
    "host": "178.156.142.230",
    "port": "5432"
}

with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        query = 'SELECT * FROM "hacker_news"."users" LIMIT 1000;'
        query2 = 'SELECT * FROM "hacker_news"."items" LIMIT 1000;'
        users = pd.read_sql_query(query, conn)
        items = pd.read_sql_query(query2, conn)

users = pd.DataFrame(users)
items = pd.DataFrame(items)
print(items.describe())
print(users.describe())
# print(test['karma'].max())
