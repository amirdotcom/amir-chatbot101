import json
import sqlite3 #for SQLite db operations
from nltk_utils import tokenize, stem, bag_of_words

# This is to load intents from intents.json
with open('intents.json', 'r') as f:
    intents = json.load(f)['intents']

# Create a connection to the SQLite database
conn = sqlite3.connect('chatbot.db')
c = conn.cursor()

# Create the 'intents' table in the database if it does not exist
c.execute('''
    CREATE TABLE IF NOT EXISTS intents (
        id INTEGER PRIMARY KEY,
        tag TEXT UNIQUE,
        patterns_id INTEGER,
        responses_id INTEGER,
        FOREIGN KEY(patterns_id) REFERENCES patterns(id),
        FOREIGN KEY(responses_id) REFERENCES responses(id)
    )
''')

# Create the 'patterns' table in the database if it does not exist
c.execute('''
    CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY,
        pattern TEXT,
        intent_id INTEGER,
        FOREIGN KEY(intent_id) REFERENCES intents(id)
    )
''')

# Create the 'responses' table in the database if it does not exist
c.execute('''
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY,
        response TEXT,
        intent_id INTEGER,
        FOREIGN KEY(intent_id) REFERENCES intents(id)
    )
''')

# This is to populate the table with data from intents.json
# Iterate through intents from the loaded JSON and populate the database
# ettttstts
for intent in intents:
    # Extract the 'tag' from the intent
    tag = intent['tag']
    # Insert or ignore the intent tag into the 'intents' table
    c.execute('INSERT OR IGNORE INTO intents (tag) VALUES (?)', (tag,))

    # Retrieve the id of the intent from the 'intents' table
    intent_id = c.execute('SELECT rowid FROM intents WHERE tag = ?', (tag,)).fetchone()
    # check if the intent_id is not none before accessing the index
    if intent_id:
        intent_id = intent_id[0]
    
    # Insert patterns
    for pattern in intent['patterns']:
        c.execute('INSERT INTO patterns (pattern, intent_id) VALUES (?, ?)', (pattern, intent_id))

    # Insert responses
    for response in intent['responses']:
        c.execute('INSERT INTO responses (response, intent_id) VALUES (?, ?)', (response, intent_id))

# Commit changes and close the connection
conn.commit()
conn.close()
