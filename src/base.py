class BasePipeline:
    """Base class for all pipeline components"""
    def __init__(self, config=None):
        self.config = config or {}
        self.initialized = False
    
    def initialize(self):
        """Initialize the component (lazy initialization)"""
        if not self.initialized:
            self._initialize()
            self.initialized = True
    
    def _initialize(self):
        """Override this method in subclasses for actual initialization"""
        pass
    
    def run(self, *args, **kwargs):
        """Main execution method"""
        self.initialize()
        return self._run(*args, **kwargs)
    
    def _run(self, *args, **kwargs):
        """Override this method in subclasses for actual execution"""
        raise NotImplementedError("Subclasses must implement _run method")