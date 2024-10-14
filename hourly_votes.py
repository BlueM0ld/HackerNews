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
    # Query to get the post titles, scores, and submission times
    query = '''
    SELECT score, time
    FROM "hacker_news"."items"
    LIMIT 1000;
    '''
    # Load the result into a DataFrame
    items = pd.read_sql_query(query, conn)

# Convert 'time' to datetime
items['time'] = pd.to_datetime(items['time'], unit='s')

# Extract the hour from the 'time' column
items['hour'] = items['time'].dt.hour

# Group by hour and calculate the average number of upvotes (score)
avg_votes_by_hour = items.groupby('hour')['score'].mean()

# Plot the average upvotes by hour
plt.figure(figsize=(10, 6))
avg_votes_by_hour.plot(kind='line', marker='o', color='skyblue')
plt.title('Average Upvotes by Hour of the Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Upvotes')
plt.grid(True)
plt.show()
