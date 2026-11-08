import json
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from team_lab import prepare, compare_time

class TeamLabTests(unittest.TestCase):
    def make(self, folder):
        path=Path(folder)/'comments.csv'
        pd.DataFrame({'body':['공연 후기']*8+[''], 'posted':['2026-09-01']*9}).to_csv(path,index=False)
        return {'input':path,'encoding':'utf-8-sig','text_column':'body','date_column':None}
    def test_optional_date_and_original_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            data,removed=prepare(self.make(folder))
            self.assertEqual(removed,1);self.assertEqual(data.source_row.tolist(),list(range(2,10)))
            self.assertTrue(data.period.eq('').all())
    def test_date_mapping_and_bad_column(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=self.make(folder);cfg['date_column']='posted'
            self.assertTrue(prepare(cfg)[0].period.eq('2026-09').all())
            cfg['text_column']='unknown'
            with self.assertRaisesRegex(ValueError,'text_column'):prepare(cfg)
    def test_no_date_never_creates_trend(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'metadata.json').write_text(json.dumps({'date_column':None}))
            compare_time({'out':out})
            self.assertFalse((out/'time_comparison.png').exists())

if __name__=='__main__':unittest.main()
