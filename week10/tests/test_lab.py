import unittest
import tempfile
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_lab import read_data, aggregate_time, audit_response

class LabTests(unittest.TestCase):
    def test_denominator_and_mixture(self):
        df=pd.DataFrame({'period':['A','A','B','B','B','B'],'T1':[.8,.2,.4,.1,.1,.2],
                         'T2':[.2,.8,.6,.9,.9,.8],'dominant_topic':['T1','T2','T2','T2','T2','T2']})
        x=aggregate_time(df,['T1','T2']).set_index(['period','topic'])
        self.assertAlmostEqual(x.loc[('A','T1'),'mean_proportion'],.5)
        self.assertAlmostEqual(x.loc[('B','T1'),'mean_proportion'],.2)
        self.assertEqual(x.loc[('B','T1'),'n_documents'],4)
        self.assertEqual(x.loc[('B','T1'),'dominant_count'],0)
    def test_citation_scope(self):
        p={'documents':[{'document_id':'D1','text':'회비를 낸 회원의 기여를 인정합니다.'}]}
        good={'claims':[{'statement':'모든 팬은 같은 생각이다.','citations':[{'document_id':'D1','quote':'회원의 기여'}]}]}
        self.assertEqual(audit_response(p,good)['status'],'HUMAN_REVIEW')
        good['claims'][0]['citations'][0]['quote']='입장이 금지됩니다'
        self.assertEqual(audit_response(p,good)['status'],'REJECT')
        good['claims'][0]['citations'][0]['document_id']='D999'
        self.assertIn('입력에 없는',audit_response(p,good)['errors'][0])
    def test_empty_and_malformed_response(self):
        p={'documents':[]}
        for response in [{},{'claims':[]},{'claims':[1]},{'claims':[{'statement':'x','citations':[]}]}]:
            self.assertEqual(audit_response(p,response)['status'],'REJECT')
    def test_ids_and_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.csv'
            df=pd.DataFrame({'document_id':[f'D{i}' for i in range(8)],'date':['2026-09-01']*8,'text':['원문']*8})
            df.to_csv(path,index=False);self.assertEqual(len(read_data(path)),8)
            df.loc[1,'document_id']='D0';df.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'중복'):read_data(path)
            df.loc[1,'document_id']='D1';df.loc[1,'date']='2026-99-99';df.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'날짜'):read_data(path)
    def test_executed_outputs(self):
        out=Path(__file__).resolve().parents[1]/'outputs'
        if not (out/'document_topics.csv').exists():self.skipTest('fit 후 실행 결과 검증')
        docs=pd.read_csv(out/'document_topics.csv');names=['T1','T2','T3','T4']
        self.assertTrue(np.allclose(docs[names].sum(axis=1),1,atol=1e-5))
        result=aggregate_time(docs,names)
        self.assertTrue(np.allclose(result.groupby('period').mean_proportion.sum(),1,atol=1e-5))
        self.assertEqual(result.groupby('period').dominant_count.sum().tolist(),docs.groupby('period').size().tolist())
if __name__=='__main__':unittest.main()
