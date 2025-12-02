SAMPLE_SCHEMA = {
    "users": {
        "columns": ["id", "name", "email"],
        "foreign_keys": []
    },
    "orders": {
        "columns": ["id", "user_id", "amount"],
        "foreign_keys": ["users"]
    }
}