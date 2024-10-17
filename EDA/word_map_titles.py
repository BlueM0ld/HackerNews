import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
        SELECT items.title
        FROM "hacker_news"."items"
        WHERE items.title IS NOT NULL
        LIMIT 1000;
        '''

        # Execute the query and load the results into a DataFrame
        result = pd.read_sql_query(query, conn)

# Display the first few rows of the result
print(result.head())
# Combine all the titles into one large string
text = ' '.join(result['title'].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot and save the word cloud to a file
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axes for the word cloud

# Save the word cloud as an image file (PNG)
wordcloud_file = 'wordcloud.png'
plt.savefig(wordcloud_file, format='png')

# Show the word cloud
plt.show()

print(f"Word cloud saved as {wordcloud_file}")
