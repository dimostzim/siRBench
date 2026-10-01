#!/usr/bin/env python3
"""Archive RPISeq-RF features from its documented public batch endpoint."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

ENDPOINT = 'http://pridb.gdcb.iastate.edu/RPISeq/batch-prot-results.php'


class ResultsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag == 'td' and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            if len(self.row) == 3 and self.row[0].startswith('>'):
                self.rows.append(self.row)
            self.row = None


def parse_probabilities(html, expected):
    parser = ResultsParser()
    parser.feed(html)
    probabilities = {}
    for identifier, rf, svm in parser.rows:
        identifier = identifier.lstrip('>')
        if identifier in probabilities:
            raise ValueError(f'Duplicate response identifier: {identifier}')
        value = float(rf)
        if not 0 <= value <= 1:
            raise ValueError(f'Invalid RF probability: {rf}')
        probabilities[identifier] = value
    if set(probabilities) != set(expected):
        raise ValueError(f'RPISeq response identifiers do not match submitted batch: received {len(probabilities)}, expected {len(expected)}')
    return probabilities


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--protein-fasta', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input_csv.open()))
    protein = ''.join(args.protein_fasta.read_text().splitlines()[1:])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    inputs = {'input_sha256': sha256(args.input_csv), 'protein_sha256': sha256(args.protein_fasta), 'endpoint': ENDPOINT, 'batch_size': 100}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['inputs'] != inputs:
            raise ValueError('Existing RPISeq archive has different inputs')
    else:
        manifest = {'inputs': inputs, 'started_utc': datetime.now(timezone.utc).isoformat(), 'batches': {}}
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    protein_path = output/'protein.sequence'
    protein_path.write_text(protein)
    for kind, column, prefix in [('siRNA','siRNA','sirna'), ('mRNA','extended_mRNA','mrna')]:
        sequences = {}
        for row in rows:
            sequence = row[column].upper().replace('T','U')
            identifier = prefix+'_'+hashlib.md5(sequence.encode()).hexdigest()[:16]
            if identifier in sequences and sequences[identifier] != sequence:
                raise ValueError('Sequence ID collision')
            sequences[identifier] = sequence
        items = sorted(sequences.items())
        all_probabilities = {}
        for offset in range(0, len(items), 100):
            batch = items[offset:offset+100]
            stem = f'{kind}_{offset//100:03d}'
            fasta = output/f'{stem}.fa'
            fasta_text = ''.join(f'>{identifier}\n{sequence}\n' for identifier, sequence in batch)
            if fasta.exists() and fasta.read_text() != fasta_text:
                raise ValueError('Existing batch inputs changed')
            fasta.write_text(fasta_text)
            response = output/f'{stem}.html'
            if response.exists():
                if stem in manifest['batches'] and sha256(response) != manifest['batches'][stem]['response_sha256']:
                    raise ValueError('Archived response checksum mismatch')
            else:
                temporary = response.with_suffix('.part')
                subprocess.run(['curl','--fail','--location','--silent','--show-error','--max-time','55',
                                '-F',f'p_input=<{protein_path}', '-F',f'r_input=@{fasta}', '-F','submit=Submit',
                                ENDPOINT, '-o',str(temporary)], check=True)
                parse_probabilities(temporary.read_text(), dict(batch))
                temporary.rename(response)
            probabilities = parse_probabilities(response.read_text(), dict(batch))
            all_probabilities.update(probabilities)
            if stem not in manifest['batches']:
                manifest['batches'][stem] = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                                           'request_sha256': sha256(fasta), 'response_sha256': sha256(response), 'rows':len(batch)}
                manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'{stem}: {len(probabilities)} complete', flush=True)
        with (output/f'{kind}_AGO2.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([kind,'RF_Classifier_prob'])
            writer.writerows(sorted(all_probabilities.items()))
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['outputs'] = {name:sha256(output/name) for name in ['siRNA_AGO2.csv','mRNA_AGO2.csv']}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
