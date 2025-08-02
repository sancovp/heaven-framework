#!/usr/bin/env python3
"""
Safe Code Reader - Strips inline comments but preserves docstrings and function signatures
"""
import sys
import ast
import re

def safe_code_read(file_path):
    """Read Python file, strip comments but keep docstrings and clean structure"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove inline comments (# at end of lines) 
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip pure comment lines (starting with #)
            if line.strip().startswith('#') and not line.strip().startswith('#!/'):
                continue
            
            # Remove inline comments but preserve strings with #
            in_string = False
            quote_char = None
            cleaned_line = ""
            i = 0
            
            while i < len(line):
                char = line[i]
                
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                
                if char == '#' and not in_string:
                    # Found comment outside string - truncate here
                    cleaned_line = cleaned_line.rstrip()
                    break
                
                cleaned_line += char
                i += 1
            
            # Only keep non-empty lines
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
        
    except Exception as e:
        return f"Error reading {file_path}: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python safe_code_reader.py <python_file>")
        sys.exit(1)
    
    result = safe_code_read(sys.argv[1])
    print(result)