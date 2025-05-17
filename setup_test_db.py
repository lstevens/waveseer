"""
Setup a test DuckDB database for Waveseer API testing.
"""

import duckdb

# Connect to the database (will create it if it doesn't exist)
db = duckdb.connect('motifs.db')

# Create patterns table
db.execute("""
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id VARCHAR PRIMARY KEY,
    label VARCHAR,
    timeframe VARCHAR,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    description TEXT,
    properties JSON
)
""")

# Insert sample patterns
sample_patterns = [
    ('hs_btc_1h_001', 'HEAD_AND_SHOULDERS', '1h', '2023-01-01', '2023-01-05', 'Head and shoulders pattern on BTC/USDT 1h', '{"confidence": 0.85}'),
    ('dt_eth_1h_001', 'DOUBLE_TOP', '1h', '2023-02-01', '2023-02-05', 'Double top pattern on ETH/USDT 1h', '{"confidence": 0.92}'),
    ('db_btc_4h_001', 'DOUBLE_BOTTOM', '4h', '2023-03-01', '2023-03-05', 'Double bottom pattern on BTC/USDT 4h', '{"confidence": 0.88}'),
    ('tr_btc_1d_001', 'TRIANGLE', '1d', '2023-04-01', '2023-04-15', 'Triangle pattern on BTC/USDT 1d', '{"confidence": 0.75}'),
    ('fl_eth_1d_001', 'FLAG', 'any', '2023-05-01', '2023-05-10', 'Flag pattern on ETH/USDT 1d', '{"confidence": 0.82}')
]

# Insert patterns
for pattern in sample_patterns:
    db.execute("""
    INSERT INTO patterns (pattern_id, label, timeframe, start_date, end_date, description, properties)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, pattern)

# Verify data
result = db.execute("SELECT * FROM patterns").fetchall()
print(f"Added {len(result)} sample patterns to database:")
for r in result:
    print(f"- {r[0]}: {r[1]} ({r[2]})")

# Close connection
db.close()

print("\nDatabase setup complete!")
