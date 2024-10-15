import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve database credentials from environment variables
hostname = os.getenv('DB_HOST')
port = os.getenv('DB_PORT', '5432')  # Default PostgreSQL port
database = os.getenv('DB_NAME')
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

# Validate that all required environment variables are set
required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
missing_vars = [var for var in required_vars if os.getenv(var) is None]
if missing_vars:
    print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)


# Create the SQLAlchemy engine
connection_string = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}'
try:
    engine = create_engine(connection_string)
    # Test the connection
    with engine.connect() as connection:
        print("Connection to PostgreSQL established successfully using SQLAlchemy.")
except Exception as e:
    print(f"Error connecting to PostgreSQL with SQLAlchemy: {e}")
    exit()

# Define your SQL query
query = """
SELECT id, time
FROM hacker_news.items 
ORDER BY items.time;
"""

# Execute the query and load data into a DataFrame
try:
    df = pd.read_sql(query, engine)
    print("Data retrieved successfully using SQLAlchemy and pandas.")
except Exception as e:
    print(f"Error executing query: {e}")
    engine.dispose()
    sys.exit(1)

# Ensure 'time' is datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['time'])

# Set 'time' as the DataFrame index
df.set_index('time', inplace=True)

# Group by month and count the number of posts
monthly_counts = df.resample('M').size().reset_index(name='post_count')

# Display the first few rows of the result
print(monthly_counts.head())

# Visualization with Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))
sns.barplot(data=monthly_counts, x='time', y='post_count', color='skyblue')
plt.title('Number of Posts Over Months', fontsize=18)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Posts', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()