import re
import pathlib

FILES = [
    'chapters/02_literature_survey.tex',
    'chapters/03_methodology.tex',
    'chapters/04_results.tex',
    'chapters/05_conclusions.tex',
]

def add_adjustbox(content):
    """Add adjustbox wrapper around tabular environments inside table environments."""
    
    def wrap_table(match):
        table_content = match.group(0)
        
        # Skip if already has adjustbox
        if 'adjustbox' in table_content:
            return table_content
        
        # Add adjustbox around tabular
        table_content = re.sub(
            r'(\\begin\{tabular\}\{[^}]+\})',
            r'\\begin{adjustbox}{max width=\\textwidth}\n    \\1',
            table_content
        )
        table_content = re.sub(
            r'(\\end\{tabular\})',
            r'\\1\n    \\end{adjustbox}',
            table_content
        )
        return table_content
    
    # Match table environments
    content = re.sub(
        r'\\begin\{table\}.*?\\end\{table\}',
        wrap_table,
        content,
        flags=re.DOTALL
    )
    return content

def main():
    for filename in FILES:
        path = pathlib.Path(filename)
        if not path.exists():
            print(f"Skipping {filename} (not found)")
            continue
            
        content = path.read_text(encoding='utf-8')
        new_content = add_adjustbox(content)
        
        if new_content != content:
            path.write_text(new_content, encoding='utf-8')
            print(f"Fixed tables in {filename}")
        else:
            print(f"No changes needed in {filename}")

if __name__ == "__main__":
    main()
