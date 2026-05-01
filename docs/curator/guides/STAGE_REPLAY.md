# Stage Replay Guide

## Overview

Stage replay is a debugging and development feature that allows you to:

1. **Save tasks** from any stage during a pipeline run
2. **Run a single stage in isolation** using saved tasks from a previous run
3. **Compare stage outputs against golden saved outputs** from a previous run

This capability can speed up iteration when developing or debugging a specific stage, without needing to re-run the entire pipeline.

**Use stage replay when:**
- 🐛 Debugging a specific stage without running the full pipeline
- 🔧 Iterating on stage logic (filtering, captioning, etc.)
- ⚡ Testing stage changes quickly with real production data
- ✅ Validating that a modified stage still matches previously saved outputs
- 📊 Profiling stage performance in isolation

**Benefits:**
- **Fast iteration**: Skip expensive upstream stages
- **Reproducibility**: Test with the exact same inputs every time
- **No code changes**: Works with any existing stage
- **Real data**: Use actual pipeline data, not synthetic test cases
- **Validation**: Compare new outputs against a known-good saved run

---

## Quick Start

### Save Tasks During a Pipeline Run

Add two flags to your pipeline command:

`--stage-save` and `--stage-save-sample-rate`.

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /path/to/videos \
  --output-clip-path /path/to/output \
  --stage-save VllmCaptionStage,MotionVectorDecodeStage \
  --stage-save-sample-rate 1.0
```

**Parameters:**
- `--stage-save`: Comma-separated list of stage class names to save input tasks from
- `--stage-save-sample-rate`: Fraction of tasks to save (0.0 = none, 1.0 = all)

**Output:**
Tasks are saved as pickle files in:
```text
{output-clip-path}/tasks/{StageName}/{video_name}_000.task.pkl
{output-clip-path}/tasks/{StageName}/{video_name}_001.task.pkl
...
```

### Run a Stage in Isolation

Once you have saved tasks, replay the stage:

Add one flag to your pipeline command: `--stage-replay`:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /path/to/videos \
  --output-clip-path /path/to/output \
  --stage-replay VllmCaptionStage \
  --limit 10
```

**Parameters:**
- `--stage-replay`: Stage class name to run in isolation, task data will be fed to this stage
- `--limit`: Optional, maximum number of task batches to load (0 = unlimited)

**What happens:**
1. Loads saved tasks from `{output-clip-path}/tasks/VllmCaptionStage/*.pkl`
2. Initializes the stage in the correct Pixi environment
3. Runs `stage.setup_on_node()` and `stage.stage_setup()`
4. Processes each task batch through `stage.process_data()`
5. Returns the output tasks

### Compare a Stage Against Golden Outputs

Once you have saved consecutive stages, compare the current stage output against the saved golden output
from the next stage:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /path/to/videos \
  --output-clip-path /path/to/output \
  --stage-compare ClipFrameExtractionStage \
  --limit 10
```

**Parameters:**
- `--stage-compare`: Stage class name, or comma-separated start/end stage names for a range
- `--stage-compare-path`: Optional alternate base path for golden outputs
- `--stage-compare-atol`: Optional numeric tolerance for numpy comparisons
- `--stage-compare-pass-threshold`: Minimum pass rate required for exit code 0
- `--limit`: Optional, maximum number of task batches to compare (0 = unlimited)

**What happens:**
1. Loads replay inputs from `{output-clip-path}/tasks/{StartStageName}/*.pkl`
2. Executes the stage or stage range directly
3. Loads golden outputs from `{base}/tasks/{NextStageName}/*.pkl`
4. Compares candidate vs golden outputs
5. Writes `report.json` to `{output-clip-path}/compare/{StartStageName}/report.json`
6. Exits nonzero if the pass rate is below the configured threshold

**Mental model:**
- `--stage-replay` = replay only
- `--stage-compare` = replay + compare + report + thresholded exit code

---

## Detailed Usage

### Workflow Example: Debugging a Caption Stage

**Scenario**: You're adding a new model to vllm_interface and want to test changes quickly without re-running the expensive video splitting stages.

#### Step 1: Run Full Pipeline and Save Tasks

First, run your full pipeline once to save inputs to the stage you want to debug:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/raw_videos \
  --output-clip-path /data/output \
  --stage-save VllmCaptionStage \
  --stage-save-sample-rate 0.1  # Save 10% of batches to reduce storage
```

**Tips:**

- Save tasks from the **same stage** you want to debug (saves inputs to that stage)
- Use `sample-rate < 1.0` for large datasets to save disk space
- The pipeline runs normally; saving happens in the background

**Check saved tasks:**

```bash
ls /data/output/tasks/VllmCaptionStage/
# video1_000.task.pkl
# video1_001.task.pkl
# video2_000.task.pkl
# ...
```

#### Step 2: Modify Your Stage

Make changes to your stage implementation:

```python
# cosmos_curator/pipelines/video/captioning/vllm_caption_stage.py

class VllmCaptionStage(CuratorStage):
    def process_data(self, tasks: list[VideoPipeTask]) -> list[VideoPipeTask]:
        # Add your experimental changes here
        logger.info(f"Processing {len(tasks)} tasks with NEW logic")

        # Your new captioning logic...

        return tasks
```

#### Step 3: Test Changes with Stage Replay

Run only the modified stage using saved tasks:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/raw_videos \
  --output-clip-path /data/output \
  --stage-replay VllmCaptionStage \
  --limit 5  # Test on just 5 batches first
```

**What's different:**

- ✅ `--stage-replay` specifies which stage to run
- ✅ `--limit` lets you test on a subset of input pickles

#### Step 4: Iterate

Repeat steps 2-3 as many times as needed:

1. Modify stage code
2. Run isolated stage with `--stage-replay`
3. Check results
4. Repeat

**No need to:**

- Re-run the full pipeline
- Re-save tasks
- Modify pipeline structure
- Add special test fixtures

### Workflow Example: Validate a Stage Change Against Golden Outputs

**Scenario**: You changed `ClipFrameExtractionStage` and want to confirm it still produces the same
output as a previous known-good run.

#### Step 1: Save Consecutive Stages

You need the saved inputs to the stage under test and the saved golden outputs from its immediate
successor:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/raw_videos \
  --output-clip-path /data/output \
  --stage-save ClipFrameExtractionStage,InternVideo2FrameCreationStage \
  --stage-save-sample-rate 1.0
```

This produces:

```text
/data/output/tasks/ClipFrameExtractionStage/
/data/output/tasks/InternVideo2FrameCreationStage/
```

The second directory contains the golden outputs for `ClipFrameExtractionStage`.

#### Step 2: Modify Your Stage

Make the stage change you want to validate.

#### Step 3: Run Stage Compare

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/raw_videos \
  --output-clip-path /data/output \
  --stage-compare ClipFrameExtractionStage \
  --limit 10
```

#### Step 4: Inspect the Result

Successful compare prints a summary like:

```text
[stage-compare] PASSED  10/10 (100.0%)  report: /data/output/compare/ClipFrameExtractionStage/report.json
```

The JSON report includes:
- overall pass/fail counts
- pass rate
- field-level summaries
- per-task failures for mismatches

---

## Finding Stage Names

Stage names must match the class name exactly. To find available stage names:

### Method 1: Check Pipeline Code

Look at the pipeline's stage instantiation:

```python
# In splitting_pipeline.py
stages = [
    CuratorStageSpec(stage=VllmPrepStage(...), max_actors=4),
    CuratorStageSpec(stage=VllmCaptionStage(...), max_actors=2),
    CuratorStageSpec(stage=ClipWriterStage(...), max_actors=8),
]

# Stage names are: VllmPrepStage, VllmCaptionStage, ClipWriterStage
```

### Method 2: Search the Codebase

```bash
# Find all stage classes
grep -r "class \w*Stage(CuratorStage)" cosmos_curator/pipelines/
```

**Common stage names:**
- **Captioning**: `VllmPrepStage`, `VllmCaptionStage`, `ApiPrepStage`, `GeminiCaptionStage`, `OpenAICaptionStage`, `EnhanceCaptionStage`
- **Filtering**: `MotionVectorDecodeStage`, `MotionFilterStage`, `AestheticsFilterStage`
- **I/O**: `ClipWriterStage`, `RemuxStage`
- **Processing**: `TranscodeStage`, `SegmentStage`, `NormalizeStage`

### Method 3: Check Saved Task Directories

If you've already saved tasks, the directory names show available stages:

```bash
ls /path/to/output/tasks/
# VllmPrepStage/
# VllmCaptionStage/
# ClipWriterStage/
```

---

## Advanced Usage

### Sampling Strategy

The `--stage-save-sample-rate` parameter uses probabilistic sampling:

```python
# Internally, for each batch:
if random.random() <= sample_rate:
    save_tasks(batch)
```

**Guidelines:**
- `1.0`: Save all batches (most accurate, largest storage)
- `0.1`: Save ~10% of batches (good for large datasets)
- `0.01`: Save ~1% of batches (quick sampling)

**Storage estimates:**
- Each `.pkl` file: ~1-10 MB (depends on task complexity)
- For 1000 videos at `sample_rate=0.1`: ~100-1000 MB
- For debugging: Start with `0.1`, increase if you need more samples

### Limiting Replay Runs

Use `--limit` to process fewer task batches:

```bash
# Test on just 1 batch
--stage-replay MyStage --limit 1

# Test on 10 batches
--stage-replay MyStage --limit 10

# Process all saved tasks (default)
--stage-replay MyStage --limit 0
```

**When to use:**
- `--limit 1`: Quick smoke test
- `--limit 10`: Reasonable sample for debugging
- `--limit 0`: Full validation before production

### Multiple Stages

Save tasks from multiple stages in one run:

```bash
--stage-save Stage1,Stage2,Stage3 --stage-save-sample-rate 0.1
```

### Compare a Stage Range

You can compare a whole stage range and validate the final output against the stage after the range:

```bash
--stage-compare Stage0,Stage1
```

This replays `Stage0` through `Stage1`, then compares the resulting output against the saved golden
tasks from the stage immediately after `Stage1` in the pipeline.

### Override the Golden Base Path

If the golden outputs live under a different run directory, override the base path:

```bash
--stage-compare ClipFrameExtractionStage \
--stage-compare-path /other/output
```

This reads replay inputs from:

```text
{output-clip-path}/tasks/ClipFrameExtractionStage/
```

and golden outputs from:

```text
/other/output/tasks/{NextStageName}/
```

### Local and S3 Paths

`--stage-save`, `--stage-replay`, and `--stage-compare` support both local paths and `s3://...`
paths through the normal `output-clip-path` / `stage-compare-path` handling.

Each stage's tasks are saved independently:
```text
output/tasks/
├── Stage1/
│   └── Stage1/*.pkl
├── Stage2/
│   └── Stage2/*.pkl
└── Stage3/
    └── Stage3/*.pkl
```

Replay any of them individually:
```bash
--stage-replay Stage2
```

### Cross-Pipeline Development

You can save tasks from one pipeline configuration and replay with different settings:

```bash
# Save tasks with original settings
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --model Qwen2-VL-2B \
  --stage-save VllmPrepStage

# Replay with experimental model
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --model Qwen2-VL-7B \
  --stage-replay VllmCaptionStage
```

**Note**: Only the replayed stage uses the new settings; saved tasks remain unchanged.

---

## Implementation Details

### How It Works

#### Saving Tasks

1. **Pipeline Build**: When `--stage-save` is provided, the pipeline wraps specified stages with `TaskSavingStage`
2. **Execution**: During `process_data()`, the wrapper probabilistically saves **input** task batches before processing
3. **Storage**: Tasks are serialized to pickle files in `{output-path}/tasks/{StageName}/`
4. **Naming**: Files are named using video name + batch index for easy identification

```python
# Simplified implementation
class TaskSavingStage(OriginalStage):
    def process_data(self, tasks):
        # Save INPUT tasks before processing
        if random.random() <= sample_rate:
            save_tasks(tasks)  # Pickle to disk
        return super().process_data(tasks)  # Then process normally
```

**Key insight**: Tasks are saved **before** the stage processes them, so `--stage-save MyStage` saves the inputs that `MyStage` will receive, not its outputs.

#### Replaying Tasks

1. **Load**: Read pickle files from `{output-path}/tasks/{StageName}/`
2. **Initialize**: Set up stage with correct Pixi environment and resources
3. **Execute**: Call `stage.process_data()` on each loaded batch
4. **Return**: Collect output tasks

```python
# Simplified implementation
def replay_stage(stage, path, limit):
    ray.init()
    stage_runner = StageRunner.options(runtime_env=pixi_env).remote(stage)
    stage_runner.stage_setup()

    for task_batch in load_tasks(path, limit):
        output_tasks.extend(stage_runner.process_data(task_batch))

    return output_tasks
```

### Environment Handling

Stage replay respects the `conda_env_name` property:

```python
class VllmCaptionStage(CuratorStage):
    @property
    def conda_env_name(self) -> str:
        return "unified"  # Replays in "unified" Pixi environment
```

This ensures GPU models, dependencies, and configurations match the original pipeline.

### Limitations

1. **No pipeline context**: Replayed stages run in isolation without access to other stages
2. **No output writing**: Summary and clip writing are skipped during replay
3. **Fixed inputs**: Saved tasks are immutable; modify the stage, not the tasks
4. **Storage required**: Pickle files can be large for complex task objects

---

## Troubleshooting

### "Stage not found in stages"

**Error:**
```text
ValueError: Stage VllmCaptionStage not found in stages
```

**Causes:**
- Typo in stage name (case-sensitive)
- Stage not present in the pipeline
- Using short name instead of class name

**Fix:**
- Check exact class name: `grep -r "class VllmCaptionStage"`

### No task files found

**Error:**
```text
# No output, or:
ls: cannot access '/path/tasks/StageName/*.pkl': No such file or directory
```

**Causes:**
- Forgot `--stage-save` during original run
- Used `sample_rate=0.0` (no tasks saved)
- Wrong `--output-clip-path`
- Stage name mismatch

**Fix:**
- Verify task directory exists: `ls {output-path}/tasks/`
- Check sample rate was > 0.0
- Ensure original run completed successfully

### Pickle deserialization errors

**Error:**
```text
pickle.UnpicklingError: invalid load key, '\x00'.
```

**Causes:**
- Task file corrupted
- Python version mismatch
- Stage class definition changed

**Fix:**
- Delete corrupted `.pkl` files
- Re-save tasks with current code
- Use same Python environment for save and replay

### Out of memory during replay

**Error:**
```text
OutOfMemoryError: Ray out of memory
```

**Causes:**
- Stage requires more GPU memory than available
- Loading too many tasks at once
- Memory leak in stage code

**Fix:**
- Use `--limit` to reduce batch count
- Check stage `resources` property: `CuratorStageResource(gpus=1)`
- Monitor memory: `watch -n 1 nvidia-smi`

### Stage behavior differs from pipeline

**Issue:** Stage produces different results when replayed vs. in pipeline.

**Possible causes:**
- Stage has hidden state from previous stages
- Stage depends on global state or external services
- Stage uses randomness without fixed seed

**Debugging:**
- Add logging to compare inputs: `logger.info(f"Tasks: {tasks[0]}")`
- Check for external dependencies (databases, APIs, files)
- Set random seeds: `random.seed(42)`, `torch.manual_seed(42)`

---

## Best Practices

### Development Workflow

1. **Save once, replay many**: One full pipeline run generates tasks for hundreds of iterations
2. **Sample appropriately**: Use `0.1` for large datasets, `1.0` for debugging specific issues
3. **Limit initially**: Start with `--limit 1` to test quickly, then increase
4. **Version control**: Commit code before replaying to track what version produced results

### Storage Management

```bash
# Check task storage usage
du -sh /path/to/output/tasks/*

# Clean up old task files
rm -rf /path/to/output/tasks/StageName/

# Save only one stage at a time to minimize storage
--stage-save MyStageThatNeedsDebugging
```

### Debugging Tips

1. **Add logging**: Stages run in Ray actors, so use `logger.info()` to see output
2. **Use debugger**: Set `RAY_DEBUG_POST_MORTEM=1` for interactive debugging
3. **Check task structure**: Load `.pkl` files manually to inspect:

```python
import pickle
with open("/path/to/tasks/Stage/Stage/video_000.task.pkl", "rb") as f:
    tasks = pickle.load(f)
    print(tasks[0].__dict__)  # Inspect first task
```

---

## Examples

### Example 1: Test Caption Changes Quickly

```bash
# Run once to save prep stage outputs
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/videos \
  --output-clip-path /data/output \
  --stage-save VllmPrepStage \
  --stage-save-sample-rate 1.0

# Iterate: Modify caption logic, then replay
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/videos \
  --output-clip-path /data/output \
  --stage-replay VllmCaptionStage \
  --limit 10

# Repeat as needed without re-running prep
```

### Example 2: Profile Stage Performance

```bash
# Save tasks
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/videos \
  --output-clip-path /data/output \
  --stage-save MotionVectorDecodeStage \
  --stage-save-sample-rate 0.2

# Profile with different configurations
time cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --output-clip-path /data/output \
  --stage-replay MotionFilterStage \
  --batch-size 16

time cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --output-clip-path /data/output \
  --stage-replay MotionFilterStage \
  --batch-size 32
```

### Example 3: A/B Test Filter Settings

```bash
# Save tasks before filter stage
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --input-video-path /data/videos \
  --output-clip-path /data/output_baseline \
  --stage-save MotionVectorDecodeStage

# Test aggressive filtering
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --output-clip-path /data/output_aggressive \
  --stage-replay MotionFilterStage \
  --motion-threshold 0.8

# Test lenient filtering  
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.splitting_pipeline \
  --output-clip-path /data/output_lenient \
  --stage-replay MotionFilterStage \
  --motion-threshold 0.3

# Compare results
python compare_outputs.py \
  --baseline /data/output_baseline/stats.json \
  --aggressive /data/output_aggressive/stats.json \
  --lenient /data/output_lenient/stats.json
```

---

## Related Documentation

- **Pipeline Design**: [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md) - Creating custom stages
- **Architecture**: [`ARCHITECTURE.md`](../reference/ARCHITECTURE.md) - Understanding Cosmos Curator structure
- **Debugging vLLM**: [`VLLM_INTERFACE_DEBUG.md`](VLLM_INTERFACE_DEBUG.md) - Debugging caption stages
- **Core Implementation**: `cosmos_curator/core/utils/misc/debug.py` - Stage replay source code

---

## Summary

Stage replay is a zero-code-change debugging tool that:

✅ **Saves tasks** from any stage during pipeline execution
✅ **Replays stages** in isolation using saved tasks
✅ **Speeds up iteration** by skipping expensive upstream stages
✅ **Works automatically** with any `CuratorStage` subclass

**Quick commands:**

```bash
# Save tasks
--stage-save StageName --stage-save-sample-rate 0.1

# Replay stage
--stage-replay StageName --limit 10
```

Start using stage replay today to iterate faster and debug more effectively!
