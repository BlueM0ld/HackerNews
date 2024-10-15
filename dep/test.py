%load_ext sql
%sql postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki
%sql select * from hacker_news.users limit 5





# Step 1: Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Define your database credentials (Not secure)
# DB_HOST = '178.156.142.230'
# DB_PORT = '5432'
# DB_NAME = 'hd64m1ki'
# DB_USER = 'sy91dhb'
# DB_PASSWORD = 'g5t49ao'

# Step 3: Load the SQL extension
%load_ext sql

# Step 4: Create the connection string
# connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Step 5: Connect to the database
# %sql $connection_string

# Step 6: Execute a SQL query and load results into a DataFrame
%sql postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki
result = %sql SELECT * FROM hacker_news.users LIMIT 5;
# df = result.DataFrame()

# Step 7: Display the DataFrame
# print(df)