"""
Proste funkcje pomocnicze do projektu.
"""

import csv
import os


def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def format_table_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def print_table(title, rows, columns=None):
    print(f"\n{title}")

    if not rows:
        print("Brak wyników.")
        return

    if columns is None:
        columns = list(rows[0].keys())

    widths = {}
    for column in columns:
        max_value_width = max(
            len(format_table_value(row.get(column, ""))) for row in rows
        )
        widths[column] = max(len(column), max_value_width)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)

    for row in rows:
        line = " | ".join(
            format_table_value(row.get(column, "")).ljust(widths[column])
            for column in columns
        )
        print(line)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def save_rows_to_csv(file_path, rows, columns=None):
    if not rows:
        return

    directory = os.path.dirname(file_path)
    if directory:
        ensure_directory(directory)

    if columns is None:
        columns = list(rows[0].keys())

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def save_text_file(file_path, content):
    directory = os.path.dirname(file_path)
    if directory:
        ensure_directory(directory)

    with open(file_path, "w", encoding="utf-8") as text_file:
        text_file.write(content)
