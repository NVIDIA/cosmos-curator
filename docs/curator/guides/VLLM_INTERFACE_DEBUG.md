# vLLM Interface Debugging Guide

## Overview

This guide helps you understand the internal code flow of the `vllm_interface` module and debug common issues. If you're just looking to **use** the interface, see [`VLLM_INTERFACE.md`](../design/VLLM_INTERFACE.md) for architecture, API reference, and configuration examples.

**Use this guide when:**
- 🐛 Debugging caption generation issues
- 🔍 Understanding how requests flow through the system
- 📚 Learning the codebase internals

**For adding new plugins, see:** [`VLLM_INTERFACE_PLUGIN.md`](VLLM_INTERFACE_PLUGIN.md)

## Quick Navigation

- [Complete Code Flow Example](#complete-code-flow-example) - Trace a single video through the system
- [Request State Lifecycle](#request-state-lifecycle) - Understanding VllmCaptionRequest states
- [Debugging Scenarios](#debugging-scenarios) - Common issues and solutions
- [Debugging Techniques](#debugging-techniques) - Where to set breakpoints, what to log

---

## Complete Code Flow Example

**Problem**: "I have to trace through 4 files to understand this"  
**Solution**: Follow this complete example of captioning one video with Qwen using the typical inflight batching path

### Scenario: Caption a single video window with two-stage refinement (typical production path)

```python
# Starting in VllmCaptionStage.process_data()
config = VllmConfig(model_variant="qwen", stage2_caption=True)
```

**Step 1**: Call main entry point
```python
# File: vllm_interface.py
captions = vllm_caption(
    model_inputs=[input_dict],  # Single Qwen-format input
    llm=llm_instance,
    processor=processor,
    sampling_params=sampling_params,
    vllm_config=config,
    max_inflight_requests=0,  # Unlimited (typical for production)
    inflight_batching=True,    # Typical production setting
    stage2_prompts=["Refine this caption..."],
)
```

**Step 2**: Dispatch to inflight batching strategy (typical path)
```python
# File: vllm_interface.py
# Since inflight_batching=True (typical), calls:
return _caption_inflight_batching(
    model_inputs, llm, processor, sampling_params, vllm_config, max_inflight_requests, stage2_prompts
)
```

**Step 3**: Create request objects and add to queue
```python
# File: vllm_interface.py
request_q.append(
    VllmCaptionRequest(
        request_id="a1b2c3d4",
        inputs={
            "prompt_token_ids": [123, 456, ...],
            "multi_modal_data": {"video": tensor}
        },
        stage2_prompt="Refine this caption...",  # From parameter
    )
)
```

**Step 4**: Submit request to vLLM engine and process continuously
```python
# File: vllm_interface.py
while len(captions) < total_requests:
    if request_q and (max_inflight_requests == 0 or len(in_flight_requests) < max_inflight_requests):
        request = request_q.popleft()
        # Submit request to vLLM's async engine
        engine.add_request(request.request_id, request.inputs, sampling_params)
        in_flight_requests[request.request_id] = request
    
    # Process one inference step (may complete multiple requests)
    engine_output = engine.step()
# Returns: [RequestOutput(request_id="a1b2c3d4", outputs=[CompletionOutput(text="A person walking")])]
```

**Step 5**: Process outputs with plugin (after engine.step() completes requests)
```python
# File: vllm_interface.py
finished = process_vllm_output(engine_output, in_flight_requests, vllm_config)

# Inside process_vllm_output
request.caption = vllm_plugin.decode(out)  # "A person walking"

# Plugin decode implementation:
# File: vllm_qwen.py
@staticmethod
def decode(vllm_output: RequestOutput) -> str:
    return vllm_output.outputs[0].text
```

**Step 6**: Handle stage 2 refinement (if needed)
```python
# File: vllm_interface.py
# Separate requests that are done vs need refinement
captions += [r.caption for r in finished if r.stage2_prompt is None]
needs_stage2 = [r for r in finished if r.stage2_prompt is not None]

# Create refinement request and ADD BACK TO QUEUE (key difference!)
for request in needs_stage2:
    refined_request = vllm_plugin.make_refined_llm_request(request, processor, request.stage2_prompt)
    request_q.append(refined_request)  # Goes back through the same engine loop

# Plugin implementation:
# File: vllm_qwen.py
@staticmethod
def make_refined_llm_request(request, processor, refine_prompt):
    # Combines refine prompt + stage 1 caption
    final_prompt = "Refine this caption..." + "A person walking"
    # Reuses original video frames
    video_frames = request.inputs["multi_modal_data"]["video"]
    # Creates new message with refinement prompt
    message = make_message(final_prompt)
    inputs = make_prompt(message, video_frames, processor)
    return VllmCaptionRequest(
        request_id="e5f6g7h8",  # New ID
        inputs=inputs,
        stage2_prompt=None,  # None = no third stage
    )
```

**Step 7**: Stage 2 request loops back through step 4
```python
# The refined request goes back into request_q 
# Next iteration: engine.add_request() submits it
# engine.step() processes it alongside any other pending requests
# Result: "A person walking along a tree-lined path in autumn"
```

**Step 8**: Return final captions when all requests complete
```python
# File: vllm_interface.py
# while loop exits when len(captions) == total_requests (original count, not including stage2)
return captions
# Returns: ["A person walking along a tree-lined path in autumn"]
```

**Key Insight**: Inflight batching allows stage 1 and stage 2 requests to be processed 
simultaneously. If you submit 10 videos, stage 2 refinement for video #1 can start while 
stage 1 for video #10 is still running. This improves GPU utilization and throughput.

### File Navigation Summary

When tracing code, you'll jump between these files in this order:

1. **vllm_interface.py** (main logic)
   - Entry: `vllm_caption()` → dispatches to batching strategy
   - Typical path: `_caption_inflight_batching()` → continuous engine.step() loop
   - Fallback path: `_caption_no_inflight_batching()` → uses `vllm_generate()` for fixed batches
   - Processing: `process_vllm_output()` → calls plugin.decode()

2. **vllm_qwen.py** (plugin implementation)
   - `decode()` → extracts text from vLLM output
   - `make_refined_llm_request()` → creates stage 2 request
   - `make_llm_input()` → converts frames to Qwen format

3. **vllm_plugin.py** (interface definition)
   - Abstract base class defining the 7 required methods
   - Only visit if you're implementing a new plugin

4. **data_model.py** (data structures)
   - `VllmConfig` → model configuration
   - `VllmCaptionRequest` → request state container

**When is no-inflight batching used?**
- Testing/debugging (simpler to step through)
- Edge cases where continuous processing isn't suitable
- Default in production: `inflight_batching=True` for better throughput

---

## Request State Lifecycle

Understanding `VllmCaptionRequest` states is key to debugging:

```python
@attrs.define
class VllmCaptionRequest:
    request_id: str                 # Unique ID for tracking
    inputs: dict[str, Any]          # Model-specific inputs
    caption: str | None = None      # Generated caption
    stage2_prompt: str | None = None  # Refinement prompt
```

### State Transitions

```text
INITIAL STATE (Stage 1 needed)
├─ caption=None
└─ stage2_prompt="Refine..." or None

    ↓ [engine.step() completes]

STAGE 1 COMPLETE
├─ caption="A person walking"
└─ stage2_prompt="Refine..." (still set)

    ↓ [make_refined_llm_request() creates new request]

STAGE 2 PENDING (new request)
├─ caption=None (reset!)
├─ stage2_prompt=None (no third stage)
└─ inputs=<new prompt with stage1 caption>

    ↓ [engine.step() completes]

FINAL STATE
├─ caption="A person walking along a tree-lined path"
└─ stage2_prompt=None
```

### Key Points

1. **`caption=None`** means "needs generation" (not "failed")
2. **`stage2_prompt=None`** means "done" (no more refinement)
3. **Stage 2 creates a NEW request** with a new `request_id`
4. **Original request count doesn't include stage 2** requests


## Debugging Scenarios

### Scenario 1: All captions are "Unknown caption"

**Symptoms:**
```python
captions = vllm_caption(...)
# Returns: ["Unknown caption", "Unknown caption", ...]
```

**Possible Cause:** Caption generation is failing, but errors are being caught.

**Debug steps:**
1. Check if vLLM model is loaded correctly:
   ```python
   print(llm.llm_engine)  # Should not be None
   ```

2. Check if requests are finishing:
   ```python
   # Add to _caption_inflight_batching or _caption_no_inflight_batching
   print(f"Finished requests: {len(finished)}, Total: {len(model_inputs)}")
   ```

3. Check vLLM output:
   ```python
   # In process_vllm_output(), before plugin.decode()
   print(f"vLLM output: {out.outputs}")
   ```

4. Check plugin decode():
   ```python
   # In your plugin's decode() method
   print(f"Decoding output: {vllm_output.outputs[0]}")
   ```

**Common causes:**
- Model path is wrong → check `VllmPlugin.model_path()`
- Inputs are malformed → check `make_llm_input()` output format
- vLLM engine error → check logs for exceptions

### Scenario 2: Stage 2 refinement isn't triggering

**Symptoms:**
```python
config = VllmConfig(stage2_caption=True, ...)
captions = vllm_caption(..., stage2_prompts=["Refine..."])
# But captions aren't refined
```

**Debug steps:**
1. Verify `stage2_prompts` is being passed:
   ```python
   # In vllm_caption()
   print(f"stage2_prompts: {stage2_prompts}")
   # Should be: ["Refine...", "Refine...", ...]
   ```

2. Check if requests have `stage2_prompt` set:
   ```python
   # In _caption_inflight_batching, after creating requests
   for req in request_q:
       print(f"Request {req.request_id}: stage2_prompt={req.stage2_prompt}")
   ```

3. Check if stage 2 filtering is working:
   ```python
   # After stage 1 completes
   print(f"Needs stage2: {len(needs_stage2)}, Done: {len([r for r in finished if r.stage2_prompt is None])}")
   ```

4. Verify plugin creates stage 2 request correctly:
   ```python
   # In make_refined_llm_request()
   print(f"Creating stage2 request. Input prompt: {final_prompt[:100]}")
   print(f"Stage2 request: caption={result.caption}, stage2_prompt={result.stage2_prompt}")
   ```

**Common causes:**
- `stage2_prompts=None` → should be list of strings
- Length mismatch → `len(stage2_prompts) != len(model_inputs)`
- Plugin doesn't implement `make_refined_llm_request()` correctly
- Forgot to set `stage2_prompt=None` in refined request

### Scenario 3: Requests are being dropped

**Symptoms:**
```python
# Submitted 100 requests, only got 95 captions back
```

**Debug steps:**
1. Track request counts:
   ```python
   # In _caption_inflight_batching
   print(f"Total requests: {total_requests}")
   print(f"Requests in queue: {len(request_q)}")
   print(f"In-flight: {len(in_flight_requests)}")
   print(f"Captions collected: {len(captions)}")
   ```

2. Check for exceptions in `process_vllm_output()`:
   ```python
   try:
       request.caption = vllm_plugin.decode(out)
   except Exception as e:
       print(f"Decode error for request {out.request_id}: {e}")
       request.caption = "Unknown caption"
   ```

3. Verify finished requests are being removed from in-flight:
   ```python
   # After processing finished requests
   print(f"Removing {len(finished)} from in_flight_requests")
   ```

4. Check loop termination condition:
   ```python
   # In while loop
   if len(captions) >= total_requests:
       print(f"Loop terminating: {len(captions)} >= {total_requests}")
   ```

**Common causes:**
- Exception in `decode()` → returns None → becomes "Unknown caption"
- Request ID mismatch → finished request not found in `in_flight_requests`
- Stage 2 requests counted in total → they shouldn't be

## Debugging Techniques

### Where to Set Breakpoints

Setting breakpoints will requires an isolated stage invocation and needs to
be run outside of the pipeline.

For maximum debugging efficiency, set breakpoints at these locations:

1. **Entry point** - `vllm_interface.py:vllm_caption()`
   - Inspect: `model_inputs`, `stage2_prompts`, `inflight_batching`
   - Verify: Lengths match, config is correct

2. **Request creation** - `_caption_inflight_batching()` after request_q.append()
   - Inspect: `request_id`, `stage2_prompt`, `inputs` keys
   - Verify: Inputs have correct structure

3. **Engine submission** - `engine.add_request()` call
   - Inspect: `request.inputs`, `sampling_params`
   - Verify: Request is submitted successfully

4. **Output processing** - `process_vllm_output()` inside the loop
   - Inspect: `engine_output`, `out.finished`, `out.outputs`
   - Verify: Outputs are completed

5. **Plugin decode** - Your plugin's `decode()` method
   - Inspect: `vllm_output.outputs[0].text`
   - Verify: Text is correct format

6. **Stage 2 creation** - `make_refined_llm_request()` in plugin
   - Inspect: `request.caption`, `refine_prompt`, result `stage2_prompt`
   - Verify: New request has `stage2_prompt=None`

### What to Log

Add logging at these critical points:

```python
from loguru import logger

# In vllm_caption()
logger.info(f"vllm_caption called with {len(model_inputs)} inputs, inflight={inflight_batching}")

# In _caption_inflight_batching()
logger.debug(f"Request queue: {len(request_q)}, In-flight: {len(in_flight_requests)}, Captions: {len(captions)}")

# In process_vllm_output()
logger.debug(f"Processing {len(engine_output)} outputs, found {len(finished)} finished")

# In plugin decode()
logger.debug(f"Decoded caption for {vllm_output.request_id}: {caption[:50]}...")

# In make_refined_llm_request()
logger.info(f"Creating stage2 request from {request.request_id}")
```

### Inspecting VllmCaptionRequest State

```python
def debug_request(request: VllmCaptionRequest, label: str):
    """Helper to print request state."""
    logger.info(f"\n=== {label} ===")
    logger.info(f"  request_id: {request.request_id}")
    logger.info(f"  caption: {request.caption[:50] if request.caption else None}...")
    logger.info(f"  stage2_prompt: {'SET' if request.stage2_prompt else 'None'}")
    logger.info(f"  inputs keys: {list(request.inputs.keys())}")
    if "multi_modal_data" in request.inputs:
        logger.info(f"  multi_modal_data keys: {list(request.inputs['multi_modal_data'].keys())}")

# Use in code:
debug_request(request, "After Stage 1")
debug_request(refined_request, "After Stage 2 Creation")
```

## References

- **Design Document**: [`VLLM_INTERFACE.md`](../design/VLLM_INTERFACE.md) - Architecture, API reference, configuration
- **Plugin Guide**: [`VLLM_INTERFACE_PLUGIN.md`](VLLM_INTERFACE_PLUGIN.md) - Step-by-step guide for adding new models
- **Profiling Guide**: [`PROFILING.md`](PROFILING.md) - Performance profiling
- **vLLM Documentation**: https://docs.vllm.ai/ - Official vLLM docs
- **Plugin Examples**: 
  - `cosmos_curator/models/vllm_qwen.py` - Most complete example
  - `cosmos_curator/models/vllm_nemotron.py` - Example with video + metadata inputs
  - `cosmos_curator/models/vllm_cosmos_reason1_vl.py` - NVIDIA model example
