# valve.py
class Valve:
    def __init__(self, size_inch, rating_class, cv_table, fl_table, xt_table, 
                 fd=1.0, d_inch=None, valve_type=3, note=""):
        """
        Initialize a valve with characteristics.
        
        Parameters:
        - size_inch: Valve size in inches
        - rating_class: Pressure rating class (150, 300, 600, etc.)
        - cv_table: Dictionary of Cv values by opening percentage
        - fl_table: Dictionary of Fl values by opening percentage
        - xt_table: Dictionary of Xt values by opening percentage
        - fd: Valve style modifier
        - d_inch: Internal diameter in inches
        - valve_type: 3 for globe, 4 for axial
        - note: Additional notes about the valve
        """
        self.size = size_inch
        self.rating_class = rating_class
        self.cv_table = cv_table
        self.fl_table = fl_table
        self.xt_table = xt_table
        self.fd = fd
        self.valve_type = valve_type  # 3: globe, 4: axial
        self.note = note
        
        if d_inch is None:
            self.diameter = size_inch  # Default: same as nominal size
        else:
            self.diameter = d_inch
        
    def get_cv_at_opening(self, open_percent: float) -> float:
        open_percent = max(0, min(100, open_percent))
        if open_percent == 0:
            return 0.0
        keys = sorted(self.cv_table.keys())
        for i in range(len(keys)-1):
            if keys[i] <= open_percent <= keys[i+1]:
                x0, x1 = keys[i], keys[i+1]
                y0, y1 = self.cv_table[x0], self.cv_table[x1]
                return y0 + (y1 - y0) * (open_percent - x0) / (x1 - x0)
        if open_percent <= keys[0]:
            return self.cv_table[keys[0]]
        return self.cv_table[keys[-1]]
    
    def get_fl_at_opening(self, open_percent: float) -> float:
        # Same implementation as original...
        open_percent = max(0, min(100, open_percent))
        if open_percent == 0:
            return 0.0
        keys = sorted(self.fl_table.keys())
        for i in range(len(keys)-1):
            if keys[i] <= open_percent <= keys[i+1]:
                x0, x1 = keys[i], keys[i+1]
                y0, y1 = self.fl_table[x0], self.fl_table[x1]
                return y0 + (y1 - y0) * (open_percent - x0) / (x1 - x0)
        if open_percent <= keys[0]:
            return self.fl_table[keys[0]]
        return self.fl_table[keys[-1]]
    
    def get_xt_at_opening(self, open_percent: float) -> float:
        # Same implementation as original...
        open_percent = max(10, min(100, open_percent))
        keys = sorted(self.xt_table.keys())
        for i in range(len(keys)-1):
            if keys[i] <= open_percent <= keys[i+1]:
                x0, x1 = keys[i], keys[i+1]
                y0, y1 = self.xt_table[x0], self.xt_table[x1]
                return y0 + (y1 - y0) * (open_percent - x0) / (x1 - x0)
        if open_percent <= keys[0]:
            return self.xt_table[keys[0]]
        return self.xt_table[keys[-1]]
