#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Audit RNAplex versus legacy ideal duplex structure')
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    args = parser.parse_args()
    with args.input_csv.open() as handle:
        records = list(csv.DictReader(handle))
    rows = []
    for row in records:
        anti = row['siRNA'].upper().replace('T', 'U')
        sense = anti.translate(str.maketrans('AUGC', 'UACG'))[::-1]
        output = subprocess.run(['RNAplex'], input=f'{sense}\n{anti}\n', text=True,
                                capture_output=True, check=True).stdout.strip()
        fields = output.split()
        structures = fields[0].split('&')
        ends = [tuple(map(int, fields[index].split(','))) for index in [1, 3]]
        full = ['.' * (start - 1) + structure + '.' * (len(anti) - end)
                for structure, (start, end) in zip(structures, ends)]
        rows.append({'record_id': row['record_id'], 'source': row['source'],
                     'secondary_structure': ' '.join(full),
                     'ideal_duplex': full == ['(' * len(sense), ')' * len(anti)]})
    with args.output_csv.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'rows': len(rows), 'nonideal_duplexes': sum(not row['ideal_duplex'] for row in rows),
                      'version': subprocess.check_output(['RNAplex', '--version'], text=True).strip()}))


if __name__ == '__main__':
    main()
