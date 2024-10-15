import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.dates as mdates
import sys
import pytz  # For timezone operations
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.error(f"Error: Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)


# Create the SQLAlchemy engine
connection_string = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}'
try:
    engine = create_engine(connection_string)
    # Test the connection
    with engine.connect() as connection:
        logging.info("Connection to PostgreSQL established successfully using SQLAlchemy.")
except Exception as e:
    logging.error(f"Error connecting to PostgreSQL with SQLAlchemy: {e}")
    sys.exit(1) 

# ---------------------------
# 2. Execute the SQL Query
# ---------------------------
# Define your SQL query
query = """
SELECT 
    DATE_TRUNC('month', time)::date AS month,
    COUNT(id) AS post_count
FROM 
    "hacker_news"."items"
GROUP BY 
    month
ORDER BY 
    month;
"""

# Execute the query and load data into a DataFrame
try:
    df = pd.read_sql(query, engine)
    logging.info("Data retrieved successfully using SQLAlchemy and pandas.")
except Exception as e:
    logging.error(f"Error executing query: {e}")
    engine.dispose()
    sys.exit(1)

# Close the engine connection
engine.dispose()


# Display the first few rows of the result
logging.info("DataFrame head:")
logging.info(df.head())
logging.info("DataFrame tail:")
logging.info(df.tail())
logging.info("\nDataFrame info:")
logging.info(df.info())

# Convert 'month' column to datetime if not already
df['month'] = pd.to_datetime(df['month'])

# Set 'month' as the DataFrame index
df.set_index('month', inplace=True)

# After executing the SQL query and loading data into DataFrame 'df'

# Verify conversion
logging.info("\nAfter datetime conversion:")
logging.info(df.dtypes)

# Set 'month' as the DataFrame index (optional, useful for time series operations)
# df.set_index('month', inplace=True)

# Ensure that 'post_count' is integer
    # df['post_count'] = df['post_count'].astype(int)

# Display the first few rows of the result
# logging.info("\nAggregated Monthly Counts:")
# logging.info(df.head())



# Create a complete date range from the first to the last month
# complete_months = pd.date_range(start=df['month'].min(), end=df['month'].max(), freq='MS')  # 'MS' stands for Month Start

# Reindex the DataFrame to include all months, filling missing post_counts with 0
# df = df.set_index('month').reindex(complete_months, fill_value=0).rename_axis('month').reset_index()

# Convert 'month' back to string format
# df['month_str'] = df['month'].dt.strftime('%b %Y')

# Ensure 'month_str' is categorical with the correct order
# df['month_str'] = pd.Categorical(df['month_str'], categories=df['month_str'], ordered=True)


# Visualization with Seaborn
sns.set_style("whitegrid")
# ---------------------------
# 3. Plot the Data
# ---------------------------

# Create a larger figure to accommodate x-axis labels
plt.figure(figsize=(30, 6))

# Create the bar plot
ax = sns.barplot(data=df, x='month', y='post_count', color='skyblue')

# Set the title and labels with increased font sizes
plt.title('Number of Posts Over Months', fontsize=20)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Number of Posts', fontsize=16)

plt.xticks(rotation=90)

# Improve x-axis date formatting
plt.gcf()

# Define the date format you want on the x-axis
# date_form = mdates.DateFormatter("%b %Y")  # e.g., 'Jan 2020'

# Set major ticks to every 6 months
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# Apply the date format to the x-axis
# ax.xaxis.set_major_formatter(date_form)


# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Display the plot
plt.show()