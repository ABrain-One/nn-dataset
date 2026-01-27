from model_selector import ModelSelector
import random

class CurriculumGenerator:
    def __init__(self, db_path="model_selection.db"):
        self.selector = ModelSelector(db_path)
        
    def generate_chain(self, start_model_id=None, length=5, min_sim=0.4, max_sim=1.0):
        """
        Generates a sequence of models where each next model is similar to the previous one,
        but arguably 'harder' or 'better' (using accuracy as a proxy for progression).
        """
        if start_model_id is None:
            start_model_id = self.selector.get_random_model_id()
            if not start_model_id:
                return []
                
        chain = [self.selector.get_model(start_model_id)]
        current_id = start_model_id
        visited = {current_id}
        
        for _ in range(length - 1):
            # Get similar models (returns list of dicts)
            candidates = self.selector.get_similar_models(current_id, min_sim, max_sim, limit=20)
            
            if not candidates:
                break
            
            # Filter visited
            candidates = [c for c in candidates if c['id'] not in visited]
            
            if not candidates:
                break
                
            # Selection Logic for "Curriculum"
            current_acc = chain[-1]['best_accuracy']
            
            # Find candidates with higher accuracy
            better_candidates = [c for c in candidates if c['best_accuracy'] > current_acc]
            
            if better_candidates:
                 # Sort by accuracy ascending (smallest improvement)
                 better_candidates.sort(key=lambda x: x['best_accuracy'])
                 next_row = better_candidates[0]
            else:
                # If no better models, pick the one with highest similarity (first in list usually, or sort?)
                # The SQL query sorts by strict similarity desc.
                next_row = candidates[0]
                
            next_model = self.selector.get_model(int(next_row['id']))
            chain.append(next_model)
            current_id = next_model['id']
            visited.add(current_id)
            
        return chain

if __name__ == "__main__":
    gen = CurriculumGenerator()
    
    # Try to find a low-accuracy model to start the chain
    # We can use the selector to find one.
    low_acc_models = gen.selector.search_models(min_acc=0.1, limit=5, order="ASC") 
    start_id = None
    if low_acc_models:
         start_id = low_acc_models[0]['id']
         print(f"Starting with Low-Accuracy Model: {low_acc_models[0]['name']} (Acc: {low_acc_models[0]['best_accuracy']:.4f})")
    
    chain = gen.generate_chain(start_model_id=start_id, length=6, min_sim=0.4, max_sim=1.0)
    
    print(f"Generated Curriculum (Length {len(chain)}):")
    for i, m in enumerate(chain):
        print(f"Step {i+1}: {m['name']} | Acc: {m['best_accuracy']:.4f}")
