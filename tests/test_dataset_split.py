"""Split integrity against real inspected metadata, plus invalid-input regressions."""
import copy
import csv
import json
import unittest
from collections import Counter
from pathlib import Path

from ml.datasets.assign_dataset_split import assign_rows, subject_assignments

REPO = Path(__file__).resolve().parents[1]


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((REPO / 'configs/dataset_split.json').read_text())
        with (REPO / 'artifacts/dataset_inspection/inventory.csv').open(newline='') as handle:
            self.rows = list(csv.DictReader(handle))

    def test_exact_frozen_membership_disjoint_and_complete(self):
        expected = {'train': [8, 4, 3, 9, 1, 2], 'validation': [10, 5], 'test': [6, 7]}
        for split, ids in expected.items():
            self.assertEqual(self.config[split], ids)
        mapping = subject_assignments(self.config)
        self.assertEqual(set(mapping), {f'Subject.{i}' for i in range(1, 11)})
        self.assertEqual(Counter(mapping.values()), {'train': 6, 'validation': 2, 'test': 2})
        sets = [set(self.config[s]) for s in expected]
        self.assertTrue(all(not a & b for i,a in enumerate(sets) for b in sets[i+1:]))

    def test_all_real_videos_assigned_once_and_metadata_preserved(self):
        result = assign_rows(self.rows, self.config)
        self.assertEqual(len(result), 100)
        self.assertEqual(len({r['relative_video_path'] for r in result}), 100)
        self.assertEqual(Counter(r['split'] for r in result), {'train':60,'validation':20,'test':20})
        counts = Counter((r['split'],r['label']) for r in result)
        for split,count in [('train',30),('validation',10),('test',10)]:
            for label in ('fall','non_fall'):
                self.assertEqual(counts[split,label], count)
        original = {r['relative_video_path']:r for r in self.rows}
        for row in result:
            self.assertEqual({k:v for k,v in row.items() if k != 'split'}, original[row['relative_video_path']])
            self.assertEqual(row['label'], 'fall' if row['activity'].startswith('Fall ') else 'non_fall')
        self.assertEqual(result, assign_rows(list(reversed(self.rows)), self.config))
        self.assertEqual(result, assign_rows(result, self.config))

    def test_reject_overlap_omissions_duplicates_and_unknown_subjects(self):
        for key, value in [('test',[1,8]),('train',[8,4,3,9,6,6]),('test',[1]),
                           ('test',[6,11])]:
            config = copy.deepcopy(self.config); config[key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                subject_assignments(config)

    def test_reject_missing_duplicate_unknown_and_conflicting_rows(self):
        cases = [self.rows[:-1], self.rows + [self.rows[0]]]
        for changes in [{'subject':'Subject.11'}, {'activity':'Other'}, {'label':'non_fall'},
                        {'split':'test'}, {'relative_video_path':'Subject.2/Fall backwards/x.avi'},
                        {'fps':'nan'}, {'readable':'False'}]:
            rows = copy.deepcopy(self.rows); rows[0].update(changes); cases.append(rows)
        for rows in cases:
            with self.subTest(first=rows[0],count=len(rows)), self.assertRaises(ValueError):
                assign_rows(rows,self.config)

    def test_final_holdout_excludes_pose_exploration(self):
        from ml.datasets.validate_pose_compatibility import SAMPLES
        report = json.loads((REPO / 'artifacts/pose_compatibility/pose_summary.json').read_text())
        sampled = {subject for subject, _ in SAMPLES}
        recorded = {int(row['subject'].split('.')[1]) for row in report['videos']}
        self.assertEqual(sampled, {1, 2, 3, 4, 5})
        self.assertEqual(recorded, sampled)
        self.assertFalse(set(self.config['test']) & sampled)
        mapping = subject_assignments(self.config)
        for subject in (6, 7):
            self.assertEqual(mapping[f'Subject.{subject}'], 'test')
        for subject in (1, 2):
            self.assertEqual(mapping[f'Subject.{subject}'], 'train')

    def test_candidate_seed_is_history_not_final_assignment(self):
        self.assertNotIn('seed', self.config)
        history = self.config['candidate_generation']
        self.assertEqual(history['seed'], 42)
        self.assertEqual(history['shuffled_subject_order'], [8,4,3,9,6,7,10,5,1,2])
        config = copy.deepcopy(self.config)
        del config['candidate_generation']
        self.assertEqual(subject_assignments(config), subject_assignments(self.config))

    def test_saved_manifest_equals_deterministic_assignment(self):
        with (REPO/'artifacts/dataset_inspection/split_inventory.csv').open(newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)),assign_rows(self.rows,self.config))


if __name__ == '__main__':
    unittest.main()
