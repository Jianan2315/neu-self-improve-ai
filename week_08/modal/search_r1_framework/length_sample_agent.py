"""Record natural rollout lengths without changing the framework generation loop."""
from search_r1_framework.agents import ScoredSearchAgent


class LengthSampleAgent(ScoredSearchAgent):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        extra = output.extra_fields['reward_extra_info']
        extra.update({
            'initial_prompt_tokens': len(output.prompt_ids),
            'trajectory_tokens': len(output.response_ids),
            'at_response_limit': int(len(output.response_ids) >= self.response_length),
            'final_answer_tokens': len(self.tokenizer.encode(extra['prediction'], add_special_tokens=False)),
            'turn_count': output.num_turns,
        })
        print(f"LENGTH_SAMPLE question={extra['question_id']} prompt={len(output.prompt_ids)} "
              f"trajectory={len(output.response_ids)} has_answer={extra['has_answer']}", flush=True)
        return output
