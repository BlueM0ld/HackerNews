# import pandas as pd
# import sqlalchemy as sa

# with engine.begin() as conn:
#     daily = pd.read_sql_query(sa.text('SELECT * FROM "hacker_news"."items" LIMIT 1;'), conn)

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
        query = 'SELECT * FROM "hacker_news"."users" LIMIT 10;'
        test = pd.read_sql_query(query, conn)
print(test)

test = pd.DataFrame(test)
print(test.head())
print(test['karma'].max())