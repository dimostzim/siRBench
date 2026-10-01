import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reconstruct_input_extents import extent_candidates, padded_window


def test_padding_and_consensus_preserve_original_unknown_bases():
    sequence = 'A'*100
    record = pd.Series({'record_id':'r','extended_mRNA':'X'+'A'*56})
    candidates = pd.DataFrame([{'reference_id':'v1','target_start_0based':30,'match_quality':'context_unknown_bases'}])
    references = pd.DataFrame([{'reference_id':'v1','molecule_type':'mRNA','accession_base':'NM1'}]).set_index('reference_id')
    result = extent_candidates(record,candidates,references,{'v1':sequence})
    assert result['context_61'][2:-2] == record.extended_mRNA
    assert result['full_mRNA'] == sequence
    assert padded_window('ACGT',-2,6) == 'XXACGTXX'


def test_conflicting_extensions_and_genomic_dna_are_not_silently_selected():
    record = pd.Series({'record_id':'r','extended_mRNA':'A'*57})
    candidates = pd.DataFrame([{'reference_id':name,'target_start_0based':21,'match_quality':'context_exact'} for name in ['v1','v2']])
    references = pd.DataFrame([{'reference_id':name,'molecule_type':'DNA','accession_base':name} for name in ['v1','v2']]).set_index('reference_id')
    result = extent_candidates(record,candidates,references,{'v1':'A'*61,'v2':'C'+'A'*60})
    assert result['context_61_status'] == 'conflicting_extensions'
    assert result['full_mRNA_status'] == 'no_confirmed_transcript'


def test_site_only_matches_do_not_supply_extended_inputs():
    record = pd.Series({'record_id':'r','extended_mRNA':'A'*57})
    candidates = pd.DataFrame([{'reference_id':'v1','target_start_0based':21,'match_quality':'target_19nt_only'}])
    result = extent_candidates(record,candidates,pd.DataFrame(),{})
    assert result['context_61_status'] == 'no_strong_match'
    assert result['full_mRNA'] == ''


def test_archived_policy_records_competing_versions_without_guessing():
    record = pd.Series({'record_id':'r','extended_mRNA':'A'*57})
    candidates = pd.DataFrame([{'reference_id':name,'target_start_0based':21,'match_quality':'context_exact'} for name in ['old','current']])
    references = pd.DataFrame([{'reference_id':'old','molecule_type':'mRNA','accession_base':'NM1','origin':'/sources/upstream/gnn/fasta'},
                               {'reference_id':'current','molecule_type':'mRNA','accession_base':'NM1','origin':'/sources/metadata.xml'}]).set_index('reference_id')
    result = extent_candidates(record,candidates,references,{'old':'A'*100,'current':'A'*120},'archived_preferred')
    assert result['full_mRNA'] == 'A'*100
    assert result['full_mRNA_selection_rule'] == 'matching_archived_transcript'
    assert result['alternative_full_sequence_count'] == 2
    assert result['available_transcript_reference_ids'] == 'current;old'
