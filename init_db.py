from ab.nn.util.db.Write import init_population
from ab.nn.util.db.Init import init_db

if __name__ == "__main__":
    init_db()
    init_population()
    print("Database re-initialized and populated.")
