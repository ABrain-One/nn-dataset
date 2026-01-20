import sqlite3

class ModelSelector:
    def __init__(self, db_path="model_selection.db"):
        self.db_path = db_path
        
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
        
    def get_model(self, model_id):
        """Fetch a single model's metadata."""
        with self._get_conn() as conn:
            # explicit column selection for clarity
            row = conn.execute("""
                SELECT id, name, task, dataset, metric, architecture, best_accuracy, total_duration_ms, path
                FROM models 
                WHERE id = ?
            """, (model_id,)).fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "name": row[1], 
                    "task": row[2],
                    "dataset": row[3],
                    "metric": row[4],
                    "architecture": row[5],
                    "best_accuracy": row[6],
                    "total_duration_ms": row[7],
                    "path": row[8]
                }
        return None

    def search_models(self, task=None, min_acc=0.0, limit=10, order="DESC"):
        """Filter models by task and accuracy."""
        query = "SELECT id, name, task, best_accuracy FROM models WHERE best_accuracy >= ?"
        params = [min_acc]
        
        if task:
            query += " AND task = ?"
            params.append(task)
            
        query += f" ORDER BY best_accuracy {order} LIMIT ?"
        params.append(limit)
        
        with self._get_conn() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            return [
                {"id": r[0], "name": r[1], "task": r[2], "best_accuracy": r[3]}
                for r in rows
            ]

    def get_similar_models(self, model_id, min_sim=0.0, max_sim=1.0, limit=10):
        """
        Finds models within a specific similarity band.
        similarity_score is between 0.0 and 1.0.
        """
        query = """
            SELECT 
                t.id, t.name, t.best_accuracy, s.similarity_score
            FROM model_similarity s
            JOIN models t ON s.target_model_id = t.id
            WHERE s.source_model_id = ?
              AND s.similarity_score BETWEEN ? AND ?
            ORDER BY s.similarity_score DESC
            LIMIT ?
        """
        params = (model_id, min_sim, max_sim, limit)
        
        with self._get_conn() as conn:
             cur = conn.execute(query, params)
             rows = cur.fetchall()
             return [
                 {"id": r[0], "name": r[1], "best_accuracy": r[2], "similarity_score": r[3]}
                 for r in rows
             ]

    def get_random_model_id(self):
        """Utility for testing."""
        with self._get_conn() as conn:
            res = conn.execute("SELECT id FROM models ORDER BY RANDOM() LIMIT 1").fetchone()
            return res[0] if res else None
