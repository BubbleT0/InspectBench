#!/usr/bin/env python3
"""Check this submission's actual build, official style, geometry and preserved data.

Run `bash BUILD.sh` first. Requires PyMuPDF for PDF inspection; the scoring tests
use the standard library. This is a file/typesetting check, not a conference
acceptance guarantee, an audit of actual model inference, or a licensing review.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
import fitz

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'author_notes/format_audit_2026-09-26'


def require(condition: bool, message: str) -> None:
    """Stop with a precise explanation when a check fails."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Return the SHA-256 of a file without altering its content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run source, compilation, PDF, and scoring checks and save their results."""
    manifest = json.loads((AUDIT / 'INPUT_MANIFEST.json').read_text())
    for path, expected in manifest['official_files'].items():
        require(digest(ROOT / path) == expected, f'Official style modified: {path}')
    for path, expected in manifest['protected_scientific_files'].items():
        require(digest(ROOT / path) == expected, f'Scientific asset modified: {path}')
    approved = manifest['approved_figure_font_only_change']
    require(digest(ROOT / approved['path']) == approved['after_sha256'],
            'The verified, font-embedded Figure 1 asset has changed.')
    maintex = (ROOT / 'main.tex').read_text()
    abstract = re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', maintex, re.S)
    require(abstract is not None, 'Abstract missing.')
    require(hashlib.sha256(abstract.group().encode()).hexdigest() == manifest['abstract_block_sha256'],
            'Abstract changed from the incoming paper.')
    sections = '\n'.join(p.read_text() for p in sorted((ROOT / 'sections').glob('*.tex')))
    tables = '\n'.join(p.read_text() for p in sorted((ROOT / 'tables').glob('*.tex')))
    source = maintex + '\n' + sections + '\n' + tables
    uncommented = '\n'.join(re.split(r'(?<!\\)%', line)[0] for line in source.splitlines())
    require(r'\iclrfinalcopy' not in uncommented, 'Camera-ready mode enabled.')
    require(not re.search(r'\\(?:geometry|newgeometry|linespread|enlargethispage|fontsize)\b', uncommented),
            'Unexpected body-layout/font override.')
    require(r'\bibliographystyle{iclr2027_conference}' in maintex, 'Wrong bibliography style.')
    require(maintex.index(r'\appendix') > maintex.index(r'\bibliography{'), 'Appendix precedes references.')
    labels = re.findall(r'\\label\{([^}]+)\}', source)
    require(len(labels) == len(set(labels)), 'Duplicate labels.')
    targets = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source))
    require(targets <= set(labels), f'Unknown reference targets: {targets-set(labels)}')
    citations = {key.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', source)
                 for key in group.split(',')}
    bbl = (ROOT / 'main.bbl').read_text()
    bibitems = set(re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', bbl, re.S))
    require(citations == bibitems and len(citations) == 48, 'Citation/bibliography mismatch.')
    require(len(re.findall(r'\\bibitem', bbl)) == 48, 'Duplicate bibliography entries.')
    for table in re.findall(r'\\begin\{table\}.*?\\end\{table\}', sections, re.S):
        before_caption = table.split(r'\caption', 1)[0]
        require(not re.search(r'\\(?:small|footnotesize|scriptsize|tiny|compacttable)\b', before_caption),
                'A table caption inherits a reduced cell font.')
    log = (ROOT / 'main.log').read_text(errors='replace')
    problems = [line for line in log.splitlines() if re.search(
        r'overfull|undefined|multiply defined|missing character:|^!|LaTeX Error', line, re.I)]
    require(not problems, 'Compilation problems: ' + '\n'.join(problems))
    require('Output written on main.pdf' in log, 'No successful PDF build.')
    aux = (ROOT / 'main.aux').read_text()
    end = re.search(r'\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}', aux)
    require(end is not None and int(end.group(1)) <= 9, 'Main text exceeds 9 pages.')
    doc = fitz.open(ROOT / 'main.pdf')
    require(doc.metadata.get('author') == 'Anonymous Authors', 'Non-anonymous PDF author metadata.')
    require(not doc.is_encrypted and doc.embfile_count() == 0, 'Encrypted PDF or embedded attachments.')
    text = '\n'.join(page.get_text() for page in doc)
    require('??' not in text and '[TBD]' not in text and 'TODO' not in text,
            'Unresolved reference or visible placeholder.')
    for heading in ['ETHICS STATEMENT', 'REPRODUCIBILITY STATEMENT', 'AI USE STATEMENT']:
        require(heading in text.upper(), f'Missing statement: {heading}')
    identity = re.compile(r'\b(?:Ziyi|Xueyuan)\b|Shanghai Jiao Tong|\bSJTU\b|/Users/|/home/|/mnt/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', re.I)
    # Bibliographic author names are legitimate third-party references; the check
    # below concerns manuscript text outside the bibliography and PDF metadata.
    text_without_refs = text.split('REFERENCES')[0] + text[text.find('ANNOTATION PROTOCOL'):]
    require(not identity.search(text_without_refs), 'Potential direct identity/path leak: inspect PDF.')
    fonts = {}
    bounds_issues = []
    for index, page in enumerate(doc, 1):
        require(abs(page.rect.width-612) < .1 and abs(page.rect.height-792) < .1,
                f'Page {index} is not US Letter.')
        require('Under review as a conference paper at ICLR 2027' in page.get_text(),
                f'Missing review header on page {index}.')
        require(any(s.isdigit() for s in page.get_text().splitlines()[:8]),
                f'Missing review line numbers on page {index}.')
        for font in page.get_fonts(full=True):
            fonts[font[0]] = {'name': font[3], 'type': font[2], 'extension': font[1]}
            require(font[1] != 'n/a' and font[2] != 'Type3', f'Unembedded/Type-3 font: {font[3]}')
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0,y0,x1,y1 = span['bbox']
                    # Exclude the official header, footer and line-number gutter.
                    if y0 > 60 and y0 < 738 and x0 > 95:
                        if x0 < 107.5 or x1 > 504.5:
                            bounds_issues.append((index,span['text'],span['bbox']))
    require(not bounds_issues, f'Horizontal text overflow: {bounds_issues[:5]}')
    linkconfig = (ROOT / 'artifact_link.tex').read_text()
    match = re.search(r'\\newcommand\{\\AnonymousArtifactURL\}\{([^}]*)\}', linkconfig)
    require(match is not None, 'Anonymous URL configuration missing.')
    url = match.group(1).strip()
    uris = [link.get('uri','') for page in doc for link in page.get_links() if link.get('uri')]
    if url:
        require(url.startswith('https://'), 'Use an HTTPS artifact URL.')
        require(url in uris, 'Configured artifact URL does not appear as a PDF link.')
    else:
        require('Anonymous benchmark resources and related code:' not in ' '.join(text.split()),
                'Empty artifact link leaves visible placeholder text.')
    require(not any('example.invalid' in uri for uri in uris), 'Test URL leaked into deliverable.')
    require(not any(p.suffix.lower() in {'.ttf','.otf','.woff','.woff2','.pfb'}
                    for p in ROOT.rglob('*') if p.is_file()), 'Standalone font files in package.')
    sys.path.insert(0, str(ROOT / 'scorer'))
    tests = unittest.defaultTestLoader.discover(str(ROOT / 'scorer'), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    require(result.wasSuccessful() and result.testsRun == 24, stream.getvalue())
    (AUDIT / 'SCORER_TESTS.txt').write_text(stream.getvalue())
    arithmetic = subprocess.run([sys.executable, str(ROOT/'scripts/audit_failure_bounds.py')],
                                cwd=ROOT, text=True, capture_output=True, check=True)
    (AUDIT / 'ARITHMETIC_TESTS.txt').write_text(arithmetic.stdout)
    require('17 formulas and 450 rounding/count corners' in arithmetic.stdout,
            'Failure-bound arithmetic checks incomplete.')
    record = {
        'input_archive':manifest['input_archive'], 'main_text_pages':int(end.group(1)),
        'total_pdf_pages':len(doc), 'official_template_files_identical':True,
        'protected_scientific_files_byte_identical':len(manifest['protected_scientific_files']),
        'abstract_byte_identical':True, 'references_cited_and_resolved':len(citations),
        'paper_size':'US Letter 612 x 792 PDF points', 'official_text_width_in':5.5,
        'official_text_height_in':9.0, 'review_headers_and_line_numbers':True,
        'font_resources_embedded':len(fonts), 'unembedded_fonts':0, 'type3_fonts':0,
        'pdf_author':'Anonymous Authors', 'direct_identity_scan_hits':0,
        'overfull_boxes':0, 'undefined_references':0, 'horizontal_text_outliers':0,
        'benign_underfull_warnings':[l for l in log.splitlines() if 'Underfull' in l],
        'blank_artifact_url':not bool(url), 'printed_artifact_placeholder':False,
        'scorer_tests_passed':result.testsRun, 'failure_bound_formulas':17,
        'failure_bound_rounding_corners':450,
        'limitations':'Does not verify OpenReview forms, author eligibility, full data/code availability, source-image permissions, model-inference correctness, or conference decisions.'
    }
    (AUDIT / 'VALIDATION.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    (AUDIT / 'PDF_FONTS.json').write_text(json.dumps(fonts, indent=2)+'\n')
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
