#### SMT runner workflow

- Entrypoint: smt_runner.py. It dynamically loads the upstream solver code from test_travelplanner.py, patches model calls to DeepSeek, patches data paths to use this repo’s database/, and writes results under smt_output/{set_type}/gpt_nl/<idx>/....
- Query loading: by default from validation_queries.json (you pass --dataset_path). Each record needs a query string.
Execution loop:
  - For each index, it creates smt_output/validation/gpt_nl/<idx>/codes/ and .../plans/.
  - Calls the upstream prompts to generate:
    - query→JSON (query.json)
    - query→steps (steps.txt)
    - steps→code (one file per section under codes/, plus aggregated codes.txt)
  - Appends the appropriate solve_{days}.txt and executes the assembled code with Z3 and local data CSVs.
  - On success (SAT), it writes plan.txt via generate_as_plan into plans/.
  - It writes time.txt with per-call timings. If no plan is produced, it now writes error.txt with a generic message; for unsat cases the upstream prints the core to stdout but doesn’t save it unless you extend it.
- Skipping/resume: --start_idx and --skip_existing let you resume long runs without redoing finished indices.
- Timeout: DeepSeek calls have a timeout (default 60s, via DEEPSEEK_TIMEOUT), so calls don’t hang indefinitely.

#### Outputs

- Per-query artifacts: smt_output/validation/gpt_nl/<idx>/
  - query.json – structured query info
  - steps.txt – constraint steps in natural language
  - *.txt – generated code per section; codes.txt concatenated
  - plan.txt – final plan if Z3 returned SAT
  - error.txt – present if no plan was written (unsat or failure)
  - time.txt – timings for each LLM/exec step
  - Aggregated output: smt_aggregated.jsonl built by aggregate_smt_output.  - py. It scans the output dirs, parses plan.txt, and emits JSONL records.
  

### commandline

```bash
% python agents/smt_runner.py \
  --set_type validation \
  --model_name "deepseek:deepseek-chat" \
  --dataset_path validation_queries.json \
  --start_idx 113 \
  --skip_existing
```

```bash
% python agents/aggregate_smt_output.py --set_type validation --output_path smt_aggregated.jsonl
```

```bash
% cd evaluation 
% python eval.py --evaluation_file_path ../smt_aggregated.jsonl
```