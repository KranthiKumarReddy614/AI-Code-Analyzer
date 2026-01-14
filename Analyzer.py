import ast
import json
import sys

class ERPNextAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {
            "methods_found": 0,
            "business_rules": [],
            "dependencies": set()
        }
        self.current_function = None

    def visit_FunctionDef(self, node):
        self.stats["methods_found"] += 1
        self.current_function = node.name
        
        # specific logic to analyze the crucial 'on_submit' method
        if node.name == 'on_submit':
            self.analyze_on_submit(node)
        
        # Continue visiting child nodes
        self.generic_visit(node)
        self.current_function = None

    def analyze_on_submit(self, node):
        """
        Drill down into the on_submit method to find what other 
        business logic it triggers.
        """
        for item in node.body:
            # Look for method calls (self.method_name())
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                if hasattr(item.value.func, 'attr'):
                    called_method = item.value.func.attr
                    
                    # Store this as a discovered business rule
                    self.stats["business_rules"].append({
                        "source_method": "on_submit",
                        "calls": called_method,
                        "line_number": item.lineno
                    })

    def visit_ImportFrom(self, node):
        """
        Track external dependencies (modules this file needs)
        """
        if node.module:
            self.stats["dependencies"].add(node.module)
        self.generic_visit(node)

def analyze_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    analyzer = ERPNextAnalyzer()
    analyzer.visit(tree)
    
    # Convert sets to lists for JSON serialization
    analyzer.stats["dependencies"] = list(analyzer.stats["dependencies"])
    
    return analyzer.stats

if __name__ == "__main__":
    # Point this to your ERPNext file
    # If you cloned erpnext, the path is likely similar to below
    target_file = "code\\erpnext\\erpnext\\accounts\\doctype\\sales_invoice\\sales_invoice.py"
    
    try:
        print(f"🔍 Analyzing {target_file}...\n")
        result = analyze_file(target_file)
        print(json.dumps(result, indent=2))
        print("\n✅ Analysis Complete.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find file at {target_file}")
        print("Please check the path in analyzer.py line 65")
