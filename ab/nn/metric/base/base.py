class BaseMetric:
    """Base class for all evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def __call__(self, outputs, targets):
        """Standard interface for all metrics - process a single batch"""
        self.update(outputs, targets)
        # Return values for compatibility with direct accumulation
        return 0, 0  # Will be overridden by subclasses
    
    def update(self, outputs, targets):
        """Update metric state with batch results"""
        pass  # Implement in subclasses
    
    def reset(self):
        """Reset the metric state"""
        pass  # Implement in subclasses
    
    def result(self):
        """Return the final metric value"""
        return 0.0  # Implement in subclasses
