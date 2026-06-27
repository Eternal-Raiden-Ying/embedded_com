#!/usr/bin/env python3
import os
import re
import json

def parse_inline(text):
    """
    Parses inline markdown format elements (bold, inline code, links) to HTML.
    """
    # Escape HTML special chars
    escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Bold **text**
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped)
    
    # Inline code `code`
    escaped = re.sub(r'`(.*?)`', r'<code class="inline-code">\1</code>', escaped)
    
    # Links [text](url) -> replace file:/// or local links to plain text/anchors
    escaped = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" class="doc-link">\1</a>', escaped)
    
    return escaped

def markdown_to_html(md_text):
    """
    Translates markdown text into clean semantic HTML.
    """
    lines = md_text.splitlines()
    html_out = []
    in_code_block = False
    in_list = False
    in_table = False
    table_headers_done = False
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                html_out.append("</code></pre></div>")
                in_code_block = False
            else:
                lang = stripped[3:].strip() or "text"
                html_out.append(f'<div class="code-block-container"><button class="copy-btn">Copy</button><pre class="code-block"><code class="language-{lang}">')
                in_code_block = True
            continue
            
        if in_code_block:
            # HTML Escape within code blocks
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_out.append(escaped)
            continue
            
        # 2. Lists
        if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("1. "):
            if not in_list:
                html_out.append('<ul class="doc-list">')
                in_list = True
            
            # Extract content
            if stripped.startswith("1. "):
                content = stripped[3:]
            else:
                content = stripped[2:]
                
            html_out.append(f"<li>{parse_inline(content)}</li>")
            continue
        elif in_list and stripped == "":
            html_out.append("</ul>")
            in_list = False
            
        # 3. Tables
        if stripped.startswith("|"):
            if not in_table:
                html_out.append('<table class="doc-table">')
                in_table = True
                table_headers_done = False
                
            # Skip divider row
            if re.match(r'^\|\s*[-:| ]+\s*\|', stripped):
                continue
                
            cols = [col.strip() for col in stripped.split("|")[1:-1]]
            html_out.append("<tr>")
            for col in cols:
                cell_content = parse_inline(col)
                if not table_headers_done:
                    html_out.append(f"<th>{cell_content}</th>")
                else:
                    html_out.append(f"<td>{cell_content}</td>")
            html_out.append("</tr>")
            
            # The first row parsed acts as the headers
            table_headers_done = True
            continue
        elif in_table and not stripped.startswith("|"):
            html_out.append("</table>")
            in_table = False
            
        # 4. Headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if header_match:
            level = len(header_match.group(1))
            h_text = parse_inline(header_match.group(2))
            html_out.append(f"<h{level} class='doc-heading h{level}'>{h_text}</h{level}>")
            continue
            
        # 5. Blockquotes
        if stripped.startswith(">"):
            html_out.append(f'<blockquote class="doc-blockquote">{parse_inline(stripped[1:].strip())}</blockquote>')
            continue
            
        # 6. Horizontal Rules
        if stripped in ("---", "***", "___"):
            html_out.append('<hr class="doc-hr">')
            continue
            
        # 7. Standard Paragraph
        if stripped:
            html_out.append(f'<p class="doc-paragraph">{parse_inline(stripped)}</p>')
            
    # Clean up unclosed structures
    if in_list:
        html_out.append("</ul>")
    if in_table:
        html_out.append("</table>")
        
    return "\n".join(html_out)

def get_markdown_files(root_dir):
    """
    Scans workspace for markdown documents.
    """
    md_files = []
    ignored_dirs = {'.git', '.pytest_cache', '__pycache__', 'backup', 'logs', 'pids', 'data'}
    
    # Root documents
    for entry in os.scandir(root_dir):
        if entry.is_file() and entry.name.endswith('.md'):
            md_files.append((os.path.abspath(entry.path), 'Root Docs'))
            
    # Subdirectories
    subdirs_mapping = {
        'docs': 'System Docs',
        'orchestrator': 'Orchestrator Docs',
        'VISTA': 'VISTA Vision Docs',
        'tools': 'Tool & Script Docs'
    }
    
    for folder, group in subdirs_mapping.items():
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for dirpath, dirnames, filenames in os.walk(folder_path):
            dirnames[:] = [d for d in dirnames if d not in ignored_dirs]
            if 'dashboard' in dirpath:
                continue
            for f in filenames:
                if f.endswith('.md'):
                    abs_path = os.path.abspath(os.path.join(dirpath, f))
                    md_files.append((abs_path, group))
                    
    return md_files

def parse_markdown(file_path, group, root_dir):
    """
    Parses metadata and content of a single markdown file.
    """
    rel_path = os.path.relpath(file_path, root_dir).replace('\\', '/')
    filename = os.path.basename(file_path)
    
    title = None
    headings = []
    summary_lines = []
    tags = [group.split(' ')[0]]
    
    encodings = ['utf-8', 'gbk', 'utf-8-sig', 'latin-1']
    content = ""
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        return {
            "id": rel_path.lower().replace('/', '_').replace('.', '_').replace('-', '_'),
            "title": filename,
            "group": group,
            "source_path": rel_path,
            "tags": tags,
            "summary": "Could not read file contents due to encoding issues.",
            "headings": [],
            "html": "<p>Encoding error reading content.</p>"
        }
        
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if not title:
            title_match = re.match(r'^#\s+(.+)$', stripped)
            if title_match:
                title = title_match.group(1).strip()
                continue
                
        heading_match = re.match(r'^(#{2,3})\s+(.+)$', stripped)
        if heading_match:
            h_level = len(heading_match.group(1))
            h_text = heading_match.group(2).strip()
            h_text_clean = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', h_text)
            headings.append({
                "level": h_level,
                "text": h_text_clean
            })
            continue
            
        if len(summary_lines) < 3 and stripped:
            if not stripped.startswith('#') and not stripped.startswith('>') and not stripped.startswith('-') and not stripped.startswith('*') and not stripped.startswith('`'):
                if len(stripped) > 5 and '=' not in stripped and ':' not in stripped:
                    summary_lines.append(stripped)
                    
    if not title:
        title = filename
        
    summary = " ".join(summary_lines)[:180]
    if summary:
        if len(summary) >= 180:
            summary += "..."
    else:
        summary = f"Smart Voice Retrieval Robot project documentation: {filename}."
        
    doc_id = rel_path.lower().replace('/', '_').replace('.', '_').replace('-', '_')
    html_content = markdown_to_html(content)
    
    return {
        "id": doc_id,
        "title": title,
        "group": group,
        "source_path": rel_path,
        "tags": tags,
        "summary": summary,
        "headings": headings,
        "html": html_content
    }

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    print(f"Scanning for Markdown documents in: {root_dir}")
    md_files = get_markdown_files(root_dir)
    print(f"Found {len(md_files)} markdown files.")
    
    manifest = []
    raw_docs = {}
    
    # Define mapping to specific dashboard module page ids
    # Mappings filter documents by their relative path
    module_mappings = {
        "orchestrator": ["orchestrator/README.md", "orchestrator/CONTROL_REFACTOR_NOTES.md"],
        "vista": ["VISTA/ReadMe.md", "VISTA/ARCHITECTURE.md", "VISTA/vision_module/STRUCTURE.md", "VISTA/vision_module/VISION_SEMANTICS_REFACTOR_NOTES.md"],
        "mobile_gateway": ["docs/mobile_gateway_runbook.md"],
        "chassis": ["ROBOT_MOTION_CONTRACT.md", "docs/docking_refactor_notes.md"],
        "arm": ["docs/ipc_refactor_notes.md"], # arm and general ipc refactors
        "cloud_grasp": ["docs/system_runbook.md"], # includes the cloud grasp pipeline info
        "startup": ["docs/system_runbook.md"],
        "config": ["docs/config.md"],
        "testing": ["docs/testing.md", "docs/offline_bag_edge_debug.md"]
    }
    
    module_pages = {k: [] for k in module_mappings.keys()}
    
    for file_path, group in md_files:
        try:
            doc_data = parse_markdown(file_path, group, root_dir)
            
            # Add to full manifest (without heavy HTML to keep manifest light)
            manifest_entry = {
                "id": doc_data["id"],
                "title": doc_data["title"],
                "group": doc_data["group"],
                "source_path": doc_data["source_path"],
                "tags": doc_data["tags"],
                "summary": doc_data["summary"],
                "headings": doc_data["headings"]
            }
            manifest.append(manifest_entry)
            
            # Add full HTML to raw_docs
            raw_docs[doc_data["id"]] = {
                "title": doc_data["title"],
                "source_path": doc_data["source_path"],
                "html": doc_data["html"]
            }
            
            # Map HTML fragments to specific module pages
            for mod_id, paths in module_mappings.items():
                if doc_data["source_path"] in paths:
                    module_pages[mod_id].append({
                        "title": doc_data["title"],
                        "source_path": doc_data["source_path"],
                        "html": doc_data["html"]
                    })
                    
            print(f"  Parsed & Compiled: {doc_data['source_path']}")
        except Exception as e:
            print(f"  Error parsing {file_path}: {e}")
            
    out_dir = os.path.join(current_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Output Manifest
    out_path_json = os.path.join(out_dir, 'docs_manifest.json')
    with open(out_path_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    out_path_js = os.path.join(out_dir, 'docs_manifest.js')
    with open(out_path_js, 'w', encoding='utf-8') as f:
        f.write("window.ROBOT_DOCS_MANIFEST = " + json.dumps(manifest, ensure_ascii=False, indent=2) + ";")
        
    # 2. Output Module Pages compiled HTML
    out_mod_json = os.path.join(out_dir, 'module_pages.json')
    with open(out_mod_json, 'w', encoding='utf-8') as f:
        json.dump(module_pages, f, ensure_ascii=False, indent=2)
    out_mod_js = os.path.join(out_dir, 'module_pages.js')
    with open(out_mod_js, 'w', encoding='utf-8') as f:
        f.write("window.ROBOT_MODULE_PAGES = " + json.dumps(module_pages, ensure_ascii=False, indent=2) + ";")
        
    # 3. Output Raw Docs compiled HTML
    out_raw_json = os.path.join(out_dir, 'raw_docs.json')
    with open(out_raw_json, 'w', encoding='utf-8') as f:
        json.dump(raw_docs, f, ensure_ascii=False, indent=2)
    out_raw_js = os.path.join(out_dir, 'raw_docs.js')
    with open(out_raw_js, 'w', encoding='utf-8') as f:
        f.write("window.RAW_DOCS = " + json.dumps(raw_docs, ensure_ascii=False, indent=2) + ";")
        
    print("Dashboard data assets compiled successfully:")
    print(f"  - Manifest: {out_path_json} & .js")
    print(f"  - Module Pages HTML: {out_mod_json} & .js")
    print(f"  - Raw Docs HTML: {out_raw_json} & .js")

if __name__ == '__main__':
    main()
