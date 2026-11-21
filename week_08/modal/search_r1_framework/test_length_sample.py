"""Check natural-sample accounting without GPU work."""
import json
from pathlib import Path
import tempfile
import unittest
from search_r1_framework.length_sample import summarize_sample


class NaturalLengthTests(unittest.TestCase):
    def test_lengths_and_limit_hits_are_not_answer_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            (output/'evaluation').mkdir()
            rows=[{'question_id':str(i),'initial_prompt_tokens':270,'trajectory_tokens':100+i,
                   'model_tokens':80+i,'tool_tokens':20,'final_answer_tokens':3,
                   'at_response_limit':0,'has_answer':1,'turn_count':4} for i in range(20)]
            path=output/'evaluation/0.jsonl'
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            selection={'question_ids':[str(i) for i in range(20)]}
            result=summarize_sample(output,selection)
            self.assertEqual(result['lengths']['trajectory_tokens']['max'],119)
            self.assertEqual(result['lengths']['final_answer_tokens']['max'],3)
            self.assertEqual(result['answers_present'],20)
            self.assertEqual(result['response_limit_hits'],0)
            rows[-1]=rows[0]
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            with self.assertRaises(ValueError):summarize_sample(output,selection)


if __name__=='__main__':unittest.main()
