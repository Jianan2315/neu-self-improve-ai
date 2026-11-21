"""Test-only veRL tool-agent subclass for actual full-length generation."""
from search_r1_framework.agents import ScoredSearchAgent
from search_r1_framework.capacity import extend_prompt, full_length_sampling


class CapacitySearchAgent(ScoredSearchAgent):
    async def run(self, sampling_params, **kwargs):
        self.capacity_stress = bool(kwargs["extra_info"].get("capacity_stress", False))
        parameters = dict(sampling_params)
        if self.capacity_stress:
            parameters = full_length_sampling(parameters, self.response_length)
        output = await super().run(parameters, **kwargs)
        if self.capacity_stress:
            if len(output.prompt_ids) != self.prompt_length or len(output.response_ids) != self.response_length:
                raise RuntimeError("Full-length generation was not reached; capacity is unverified.")
            print(f"CAPACITY_LENGTH_OK prompt={len(output.prompt_ids)} response={len(output.response_ids)}", flush=True)
        return output

    async def _handle_pending_state(self, agent_data, sampling_params):
        state = await super()._handle_pending_state(agent_data, sampling_params)
        if self.capacity_stress:
            # Use a real non-padding token with attention enabled for every position.
            fill = self.tokenizer.encode(" context", add_special_tokens=False)[0]
            agent_data.prompt_ids = extend_prompt(agent_data.prompt_ids, self.prompt_length, fill)
        return state
