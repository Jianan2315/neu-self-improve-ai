"""Thin scoring adapters; generation, tool calls and masks are veRL's implementation."""
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from search_r1_framework.scoring import last_generated_segment, score_final


class OutcomeScoring:
    async def run(self, sampling_params, **kwargs):
        # The entire rollout is executed by the existing framework agent.
        output = await super().run(sampling_params, **kwargs)
        ids = last_generated_segment(output.response_ids, output.response_mask)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        result = score_final(text, kwargs["reward_model"]["ground_truth"]["target"])
        output.reward_score = result["score"]
        output.extra_fields["reward_extra_info"] = {
            **result, "question_id": kwargs["extra_info"]["question_id"],
            "model_tokens": int(sum(output.response_mask)),
            "tool_tokens": int(len(output.response_mask) - sum(output.response_mask)),
        }
        return output


class ScoredSearchAgent(OutcomeScoring, ToolAgentLoop):
    """Existing multi-turn tool agent plus assignment outcome scoring."""


class ScoredDirectAgent(OutcomeScoring, SingleTurnAgentLoop):
    """Existing single-turn agent plus the same outcome scoring."""
