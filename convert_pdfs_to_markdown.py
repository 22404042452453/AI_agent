#!/usr/bin/env python3
"""
Script to convert PDF files to markdown format for improved RAG processing
"""

import os
import sys
import re
from pathlib import Path
from markitdown import MarkItDown
import pdfplumber

def extract_text_with_pdfplumber(pdf_path):
    """
    Extract text from PDF using pdfplumber for better formula handling

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content
    """
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n\n"
    except Exception as e:
        print(f"Warning: pdfplumber extraction failed: {e}")
        return ""

    return text_content

def fix_corrupted_formulas(text):
    """
    Attempt to fix corrupted mathematical formulas in the text

    Args:
        text (str): Text with potentially corrupted formulas

    Returns:
        str: Text with fixed formulas where possible
    """
    # Common formula corruption patterns and their fixes
    fixes = [
        # Fix corrupted Greek letters and symbols
        (r'q>', 'φ'),  # q> → φ (phi)
        (r'cosq>', 'cosφ'),  # cosq> → cosφ
        (r'sinq>', 'sinφ'),  # sinq> → sinφ
        (r'tgq>', 'tgφ'),  # tgq> → tgφ

        # Fix corrupted subscripts - common patterns
        (r'(.)\s*фак1', r'\1_факт'),  # (.фак1 → (_факт
        (r'(.)\s*ном', r'\1_ном'),  # (.ном → (_ном
        (r'(.)\s*пр', r'\1_пр'),  # (.пр → (_пр
        (r'(.)\s*рф', r'\1_рф'),  # (.рф → (_рф
        (r'(.)\s*нас', r'\1_нас'),  # (.нас → (_нас
        (r'(.)\s*макс', r'\1_макс'),  # (.макс → (_макс
        (r'(.)\s*мин', r'\1_мин'),  # (.мин → (_мин
        (r'(.)\s*факт', r'\1_факт'),  # (.факт → (_факт

        # Fix corrupted superscripts
        (r'/,\s*\^\s*', '₁'),  # /, ^ → ₁
        (r'ы>ы', '₂'),  # ы>ы → ₂
        (r'(.)\s*2', r'\1₂'),  # (.2 → (_2
        (r'(.)\s*3', r'\1₃'),  # (.3 → (_3

        # Fix corrupted operators and symbols
        (r'\^\s*и\s*А\s*акт', '_нагрузки'),  # ^иАакт → _нагрузки
        (r'\^\s*и\s*А\s*акт', '_нагрузки'),  # Alternative pattern
        (r'(.)\s*\'\s*=\s*2\s*4\.\s*\+\s*2\s*i\s*W', r'\1\' = 2R_пр + 2R_рф'),  # W'=24.+2iW → W' = 2R_пр + 2R_рф

        # Fix specific corrupted formula patterns
        (r'А\s*\*\s*\*\s*А', 'A * * A'),  # Keep as is but clean up
        (r'2\s*г\s*пр\s*\+\s*2\s*г\s*рф', '2R_пр + 2R_рф'),  # 2гпр+2грф → 2R_пр + 2R_рф

        # Fix corrupted variable names
        (r'Н\s*ноы', 'I₂ном'),  # Н ноы → I₂ном
        (r'SM\s*ном', 'S_ном'),  # SM ном → S_ном
        (r'Т\s*Т', 'ТТ'),  # Т Т → ТТ
        (r'К\s*ном', 'К_ном'),  # К ном → К_ном
        (r'Я-2', 'R₂'),  # Я-2 → R₂
        (r'Х2', 'X₂'),  # Х2 → X₂
        (r'Я2', 'R₂'),  # Я2 → R₂

        # Keep common abbreviations as is
        (r'ТТ', 'ТТ'),  # Keep ТТ as is
        (r'КЗ', 'КЗ'),  # Keep КЗ as is
        (r'ВАХ', 'ВАХ'),  # Keep ВАХ as is
        (r'ПХН', 'ПХН'),  # Keep ПХН as is
        (r'ЭДС', 'ЭДС'),  # Keep ЭДС as is
        (r'ЛЭП', 'ЛЭП'),  # Keep ЛЭП as is
        (r'ОРУ', 'ОРУ'),  # Keep ОРУ as is
    ]

    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)

    return text

def convert_pdf_to_markdown(pdf_path, output_dir=None):
    """
    Convert a single PDF file to markdown

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Output directory. If None, saves in same directory as PDF

    Returns:
        str: Path to the created markdown file, or None if conversion failed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        return None

    # Determine output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = pdf_path.parent

    # Create markdown filename
    markdown_filename = pdf_path.stem + ".md"
    markdown_path = output_dir / markdown_filename

    try:
        print(f"Converting {pdf_path.name} to markdown...")

        # Try pdfplumber first for better text extraction
        markdown_content = extract_text_with_pdfplumber(str(pdf_path))

        if not markdown_content or len(markdown_content.strip()) < 1000:
            # Fallback to MarkItDown if pdfplumber didn't work well
            print("  Using MarkItDown as fallback...")
            markitdown = MarkItDown()
            result = markitdown.convert(str(pdf_path))
            markdown_content = result.text_content

        # Post-process to fix corrupted formulas
        markdown_content = fix_corrupted_formulas(markdown_content)

        # Basic markdown formatting
        # Add headers for sections that look like they should be headers
        lines = markdown_content.split('\n')
        formatted_lines = []
        for line in lines:
            # Convert lines that look like section headers
            if re.match(r'^\d+\.?\d*\.?\d*\s+[А-ЯA-Z]', line.strip()):
                # This looks like a section header, make it a markdown header
                formatted_lines.append('## ' + line.strip())
            else:
                formatted_lines.append(line)

        markdown_content = '\n'.join(formatted_lines)

        # Save markdown file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"✓ Converted: {pdf_path.name} → {markdown_filename}")
        print(f"  Content length: {len(markdown_content)} characters")

        return str(markdown_path)

    except Exception as e:
        print(f"✗ Failed to convert {pdf_path.name}: {e}")
        return None

def convert_directory_pdfs(input_dir, output_dir=None, recursive=True):
    """
    Convert all PDF files in a directory to markdown

    Args:
        input_dir (str): Input directory containing PDF files
        output_dir (str, optional): Output directory for markdown files
        recursive (bool): Whether to process subdirectories recursively
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory not found: {input_path}")
        return

    # Find all PDF files
    if recursive:
        pdf_files = list(input_path.rglob("*.pdf"))
    else:
        pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to convert")

    converted_count = 0
    failed_count = 0

    for pdf_file in pdf_files:
        result = convert_pdf_to_markdown(pdf_file, output_dir)
        if result:
            converted_count += 1
        else:
            failed_count += 1

    print("\nConversion summary:")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(pdf_files)}")

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python convert_pdfs_to_markdown.py <input_directory> [output_directory]")
        print("Example: python convert_pdfs_to_markdown.py files files_markdown")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    convert_directory_pdfs(input_dir, output_dir)

if __name__ == "__main__":
    main()
