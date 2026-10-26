import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_lda as lab

class TestLDA(unittest.TestCase):
    def test_full_run_and_distributions(self):
        with tempfile.TemporaryDirectory() as d:
            out=lab.main(['--tokenizer','regex','--compare','2','4','--max-iter','3','--output-root',d])
            for k in [2,4]:
                theta=pd.read_csv(out/f'k{k}/document_topics.csv').filter(regex=r'^topic_')
                phi=pd.read_csv(out/f'k{k}/topic_word_distribution.csv').filter(regex=r'^topic_')
                self.assertTrue(np.allclose(theta.sum(axis=1),1))
                self.assertTrue(np.allclose(phi.sum(axis=0),1))
                self.assertTrue((out/f'k{k}/report.html').exists())
            self.assertTrue((out/'run_metadata.json').exists())
    def test_clean_and_invalid_column(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'input.csv'
            pd.DataFrame({'review':['a','b','c','d','e','a',' ']}).to_csv(f,index=False)
            clean,excluded=lab.read_documents(f,'review')
            self.assertEqual(len(clean),5)
            self.assertEqual(set(excluded.reason),{'empty_text','duplicate_text'})
            with self.assertRaises(ValueError):lab.read_documents(f,'text')
    def test_html_escape_and_csv_formula(self):
        frame=pd.DataFrame({'text':['<script>alert(1)</script>','=1+1']})
        self.assertNotIn('<script>',lab.table(frame))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'safe.csv';lab.safe_csv(frame,f)
            self.assertEqual(pd.read_csv(f).text.iloc[1],"'=1+1")
    def test_reproducible_model(self):
        docs,_,counts,words,_=lab.prepare(lab.ROOT/'data/culture_reviews.csv','text','regex',lab.ROOT/'data/stopwords.txt',2,.95)
        a=lab.fit_topics(counts,words,docs,4,42,.1,.1,3)
        b=lab.fit_topics(counts,words,docs,4,42,.1,.1,3)
        self.assertTrue(np.allclose(a[4],b[4]))

if __name__=='__main__':unittest.main()
