import sys
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lab

class LabTest(unittest.TestCase):
    def test_data_and_metrics(self):
        train, probe=lab.load_data()
        self.assertEqual(len(probe),16)
        self.assertFalse(set(train.text)&set(probe.text))
        result,_=lab.run_baselines(train,probe)
        result['llm']=-1
        scores=lab.compare_metrics(result).set_index('method')
        self.assertEqual(scores.loc['llm','n'],12)
        self.assertEqual(scores.loc['llm','accuracy_all'],0)
        self.assertEqual(scores.loc['llm','macro_f1'],0)
        self.assertEqual(scores.loc['llm','binary_coverage'],0)

    def test_schema_and_evidence(self):
        text='재미있지 않다.'
        self.assertEqual(lab.parse_output('{"label":"부정","evidence":"재미있지 않다"}',text)['llm'],0)
        self.assertEqual(lab.parse_output('```json\n{}\n```',text)['status'],'invalid_json')
        self.assertEqual(lab.parse_output('{"label":"부정","evidence":"없는 말"}',text)['status'],'invalid_evidence')
        self.assertEqual(lab.parse_output('{"label":"positive","evidence":"재미있지 않다"}',text)['status'],'invalid_label')

    def test_no_labels_or_keys_in_prompt_payload(self):
        captured={}
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"긍정","evidence":"재미있다"}'),finish_reason='stop')],model=lab.MODEL_ID,usage=SimpleNamespace(total_tokens=20))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        answer=lab.classify_one(client,'재미있다')
        payload=json.loads(captured['messages'][1]['content'])
        self.assertEqual(set(payload),{'target','review'})
        self.assertNotIn('gold_label',json.dumps(captured,ensure_ascii=False))
        self.assertEqual(answer['status'],'ok')
        self.assertEqual(captured['extra_body']['chat_template_kwargs']['enable_thinking'],False)

if __name__=='__main__':unittest.main()
