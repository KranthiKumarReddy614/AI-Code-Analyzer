import ast
from pathlib import Path
from typing import Dict, List, Any
import json


def extract_entities_from_directory(path: Path, verbose: bool = False) -> Dict[str, List[Dict]]:
    """
    Extract entities from all Python files in a directory.

    Args:
        path: Directory or file path to analyze
        verbose: If True, print progress

    Returns:
        Dictionary mapping file paths to list of entities found

    Example return value:
        {
            "sales_invoice.py": [
                {"type": "class", "name": "SalesInvoice", "line": 10, ...},
                {"type": "function", "name": "validate", "line": 50, ...},
            ]
        }
    """
    all_entities = {}

    # Handle single file vs directory
    if path.is_file():
        python_files = [path] if path.suffix == ".py" else []
    else:
        # rglob finds files recursively (all subdirectories)
        python_files = list(path.rglob("*.py"))

    if verbose:
        print(f"Found {len(python_files)} Python files")

    for file_path in python_files:
        try:
            entities = extract_entities_from_file(file_path)
            if entities:  # Only include files with entities
                # Use relative path for cleaner output
                relative_path = str(file_path.relative_to(path.parent))
                all_entities[relative_path] = entities

                if verbose:
                    print(f"  {relative_path}: {len(entities)} entities")

        except Exception as e:
            # Don't crash on one bad file - log and continue
            if verbose:
                print(f"  Error parsing {file_path}: {e}")

    return all_entities


def extract_entities_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract all entities from a single Python file.

    This is where the AST magic happens:
    1. Read the source code
    2. Parse it into an AST
    3. Walk the tree and collect entities

    Args:
        file_path: Path to the Python file

    Returns:
        List of entity dictionaries
    """
    # Read the source code
    source_code = file_path.read_text(encoding="utf-8")

    # Parse into AST
    # This can fail on syntax errors - that's why we catch exceptions above
    tree = ast.parse(source_code, filename=str(file_path))

    entities = []

    # Walk ONLY the top level first (not nested)
    # ast.walk() visits ALL nodes, but we want to handle nesting ourselves
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Found a class!
            class_entity = extract_class(node, source_code)
            entities.append(class_entity)

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Found a top-level function!
            func_entity = extract_function(node, source_code, parent_class=None)
            entities.append(func_entity)

    return entities


def extract_class(node: ast.ClassDef, source_code: str) -> Dict[str, Any]:
    """
    Extract information about a class definition.

    We collect:
    - Name
    - Line number (for navigation)
    - Base classes (what it inherits from)
    - Decorators (like @dataclass)
    - Methods (functions defined inside)
    - Docstring (if present)

    Args:
        node: The AST ClassDef node
        source_code: Original source (for getting docstrings)

    Returns:
        Dictionary with class information
    """
    # Get the class docstring
    # ast.get_docstring() handles the triple-quote parsing for us
    docstring = ast.get_docstring(node) or ""

    # Get base classes (inheritance)
    # e.g., class Foo(Bar, Baz) -> bases are ["Bar", "Baz"]
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            # For things like models.Model
            bases.append(f"{get_attribute_name(base)}")

    # Get decorators
    # e.g., @dataclass, @frappe.whitelist()
    decorators = [get_decorator_name(d) for d in node.decorator_list]

    # Extract methods (functions inside the class)
    methods = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
            method = extract_function(item, source_code, parent_class=node.name)
            methods.append(method)

    return {
        "type": "class",
        "name": node.name,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "bases": bases,
        "decorators": decorators,
        "docstring": docstring[:200] if docstring else "",  # Truncate long docstrings
        "methods": methods,
        "method_count": len(methods),
    }


def extract_function(node, source_code: str, parent_class: str = None) -> Dict[str, Any]:
    """
    Extract information about a function/method definition.

    We collect:
    - Name
    - Line number
    - Parameters (arguments)
    - Decorators
    - Docstring
    - Whether it's async
    - Parent class (if it's a method)

    Args:
        node: The AST FunctionDef or AsyncFunctionDef node
        source_code: Original source code
        parent_class: Name of parent class if this is a method

    Returns:
        Dictionary with function information
    """
    # Get docstring
    docstring = ast.get_docstring(node) or ""

    # Get parameter names
    # node.args contains: args, posonlyargs, kwonlyargs, defaults, etc.
    params = []
    for arg in node.args.args:
        params.append(arg.arg)

    # Get decorators
    decorators = [get_decorator_name(d) for d in node.decorator_list]

    # Check if it's async
    is_async = isinstance(node, ast.AsyncFunctionDef)

    # Determine the "kind" of function
    if parent_class:
        if node.name == "__init__":
            kind = "constructor"
        elif node.name.startswith("_"):
            kind = "private_method"
        else:
            kind = "method"
    else:
        kind = "function"

    return {
        "type": "function",
        "kind": kind,
        "name": node.name,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "params": params,
        "decorators": decorators,
        "docstring": docstring[:200] if docstring else "",
        "is_async": is_async,
        "parent_class": parent_class,
    }


def get_decorator_name(node) -> str:
    """
    Get the name of a decorator.

    Decorators can be:
    - Simple: @staticmethod -> Name node
    - Attribute: @app.route -> Attribute node
    - Call: @pytest.mark.skip(reason="...") -> Call node

    Args:
        node: The decorator AST node

    Returns:
        String representation of the decorator
    """
    if isinstance(node, ast.Name):
        # @staticmethod
        return node.id
    elif isinstance(node, ast.Attribute):
        # @app.route
        return get_attribute_name(node)
    elif isinstance(node, ast.Call):
        # @decorator(args) - recurse to get the function being called
        return get_decorator_name(node.func)
    else:
        return "unknown_decorator"


def get_attribute_name(node) -> str:
    """
    Get the full name of an attribute access.

    For `foo.bar.baz`, this returns "foo.bar.baz"

    Args:
        node: The Attribute AST node

    Returns:
        Dot-separated name string
    """
    parts = []
    current = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)

    # Parts are collected in reverse order
    return ".".join(reversed(parts))


def main():
    # 🔴 CHANGE THIS PATH to where you cloned ERPNext
    ERPNEXT_PATH = Path(r"code/erpnext/erpnext")

    if not ERPNEXT_PATH.exists():
        raise FileNotFoundError(f"Path not found: {ERPNEXT_PATH}")

    print("Starting ERPNext extraction...")
    entities = extract_entities_from_directory(
        ERPNEXT_PATH,
        verbose=True
    )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "erpnext_entities.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2)

    print(f"\n✅ Extraction complete")
    print(f"📄 Output saved to: {output_file}")
    print(f"📦 Files analyzed: {len(entities)}")


if __name__ == "__main__":
    main()