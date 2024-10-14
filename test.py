import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

# Database connection parameters
conn_params = {
    "dbname": "hd64m1ki",
    "user": "sy91dhb",
    "password": "g5t49ao",
    "host": "178.156.142.230",
    "port": "5432"
}

# Connect to the database
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        # Query to get the post IDs and their submission times
        query = '''
        SELECT items.time
        FROM "hacker_news"."items" 
        LIMIT 9000000;
        '''

        # Execute the query and load the results into a DataFrame
        result = pd.read_sql_query(query, conn)

# Display the first few rows of the result
print(result.head())

# Convert the time column to datetime
result['time'] = pd.to_datetime(result['time'])

# Group by month and count the number of posts
result.set_index('time', inplace=True)
monthly_counts = result.resample('M').size()

# Plotting the number of posts over months
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Posts Over Months')
plt.xlabel('Month')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
