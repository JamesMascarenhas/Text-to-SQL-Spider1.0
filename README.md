Text-to-SQL on Spider 1.0

Code for every experiment in the report:

- **Task 1:** a zero-shot benchmark of 11 configurations of Qwen2.5-Coder and Qwen3 on the full Spider 1.0 development split (1,034 questions).
- **Task 2:** an analysis of the Qwen3 reasoning traces.
- **Task 3:** execution-based self-consistency (EBSC) for Qwen3-4B with thinking.

The evaluation set is the **whole Spider development split**, so there is no random subset to reproduce. `outputs/subset_ids.json` lists all 1,034 example ids after step 2 below.

Only model inference (step 3) needs a GPU. Everything else runs on a normal computer, and every local step below was rerun from a clean clone. It reproduced the original prompts, scores and summary tables exactly.

---

## 1. Requirements

| Part | Needs |
|---|---|
| Local steps (prompts, scoring, statistics, figures) | Python **3.11 to 3.13** (we used 3.12.7 on macOS), about 2 GB of disk for Spider |
| Inference | A CUDA GPU with **at least 24 GB of memory** (we used one NVIDIA L4 on Google Colab Pro). Smaller GPUs can run the small models, but Qwen3-8B needs about 24 GB |
| Internet | To download Spider, the evaluator, the NLTK tokenizer and the model weights from the Hugging Face Hub |

All models are public, so **no Hugging Face access token is needed**. If you want to use one anyway, set it with `huggingface-cli login` or the `HF_TOKEN` environment variable, never in the code.

---

## 2. Local setup

Run everything from the repository root (the folder that contains `src/`).

### 2.1 Python environment

```bash
python3 -m venv venv
source venv/bin/activate                 # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt_tab      # tokenizer used by the official Spider parser
```

### 2.2 Spider 1.0 data

Download `spider_data.zip` from the official Spider page (https://yale-lily.github.io/spider, "Spider Dataset" link) and unzip it into `data/`. The scripts expect exactly this layout:

```
data/spider_data/dev.json
data/spider_data/tables.json
data/spider_data/database/<db_id>/<db_id>.sqlite      (166 databases; the 20 dev databases are used)
```

### 2.3 Official evaluator

The test suite evaluator (Zhong et al., 2020) is used for EM, EX, component F1 and the difficulty levels. Clone it at the commit we used:

```bash
git clone https://github.com/taoyds/test-suite-sql-eval eval/test-suite-sql-eval
git -C eval/test-suite-sql-eval checkout e97acc546ecbee8fa27fa8dbf025ef61493a876c
```

You should now have:

```
data/spider_data/...
eval/test-suite-sql-eval/evaluation.py
src/...
```

### 2.4 Check the setup

```bash
python src/prepareSubset.py --n 1034
python src/buildPrompts.py
python src/sanityCheck.py
```

1. `prepareSubset.py --n 1034` labels every development question with its Spider difficulty and writes `outputs/subset.jsonl` and `outputs/subset_ids.json`. Seed 42 is fixed in `src/spiderUtils.py`, and with `--n 1034` the whole split is used.
2. `buildPrompts.py` writes `outputs/prompts.jsonl`, the zero-shot prompt with the `CREATE TABLE` schema and no example rows (1,034 prompts).
3. `sanityCheck.py` takes about 2 minutes and must end with `all sanity checks passed`. It checks SQL extraction, checks that gold queries scored against themselves give 100% on EM and EX with both our scorer and the official script, and checks that empty outputs are scored as wrong.

---

## 3. Inference (GPU)

### 3.1 Google Colab setup (what we used)

In a Colab notebook with an **L4 GPU** runtime:

```bash
!git clone <(https://github.com/JamesMascarenhas/cisc839-a1)> a1      # or upload and unzip the submitted code folder
%cd a1
!pip install -q -r requirements-gpu.txt    # installs vllm 0.30.0, which brings torch and transformers
```

Then put the prompts in place. Either option works:

- **Option A (simplest):** upload the `outputs/prompts.jsonl` built locally in step 2.4 to `a1/outputs/prompts.jsonl`. Inference only needs this one file.
- **Option B:** repeat steps 2.2 to 2.4 inside Colab. Install `requirements.txt` only for this, not together with vLLM.

### 3.2 Run the models

Each command writes `outputs/runs/<name>.jsonl` (one row per question with the raw output, the extracted SQL, token counts and time) and `outputs/runs/<name>_config.json` (settings, exact revision, GPU and library versions). Keep the run names exactly as written: the later scripts look for these names.

**Pinned revisions** (the same as Table 1 and Appendix B of the report):

```
Qwen/Qwen2.5-Coder-0.5B-Instruct       ea3f2471cf1b1f0db85067f1ef93848e38e88c25
Qwen/Qwen2.5-Coder-1.5B-Instruct       2e1fd397ee46e1388853d2af2c993145b0f1098a
Qwen/Qwen2.5-Coder-3B-Instruct         488639f1ff808d1d3d0ba301aef8c11461451ec5
Qwen/Qwen2.5-Coder-7B-Instruct         c03e6d358207e414f1eca0bb1891e29f1db0e242
Qwen/Qwen2.5-Coder-7B-Instruct-AWQ     8e8ed243bbe6f9a5aff549a0924562fc719b2b8a
Qwen/Qwen3-1.7B                        70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
Qwen/Qwen3-4B                          1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B                          b968826d9c46dd6066d109eabc6255188de91218
```

**Decoding** is set by `--thinking`:

- `none` and `off` use greedy decoding (temperature 0, top-p 1, no top-k).
- `on` uses temperature 0.6, top-p 0.95 and top-k 20, from the Qwen3 model card.

Batches are 100 prompts each, the vLLM context window is 16,384 tokens (10,240 for Qwen3-8B so it fits in 24 GB), and the seed is 42 unless `--seed` is given.

```bash
# Qwen2.5-Coder (no reasoning), greedy, up to 512 new tokens
python src/runInference.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25 --name full-qwen25coder-0.5b --thinking none
python src/runInference.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a --name full-qwen25coder-1.5b --thinking none
python src/runInference.py --model Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --name full-qwen25coder-3b --thinking none
python src/runInference.py --model Qwen/Qwen2.5-Coder-7B-Instruct --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --name full-qwen25coder-7b --thinking none
python src/runInference.py --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --revision 8e8ed243bbe6f9a5aff549a0924562fc719b2b8a --name full-qwen25coder-7b-awq --thinking none --dtype auto

# Qwen3 with thinking off, greedy, up to 512 new tokens
python src/runInference.py --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --name full-qwen3-1.7b-nothink --thinking off
python src/runInference.py --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --name full-qwen3-4b-nothink --thinking off
python src/runInference.py --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --name full-qwen3-8b-nothink --thinking off --maxModelLen 10240

# Qwen3 with thinking on, sampled, up to 8,192 new tokens
python src/runInference.py --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --name full-qwen3-1.7b-think --thinking on --maxNewTokens 8192
python src/runInference.py --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --name full-qwen3-1.7b-think-seed66 --thinking on --maxNewTokens 8192 --seed 66
python src/runInference.py --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --name full-qwen3-1.7b-think-seed73 --thinking on --maxNewTokens 8192 --seed 73
python src/runInference.py --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --name full-qwen3-4b-think --thinking on --maxNewTokens 8192
python src/runInference.py --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --name full-qwen3-4b-think-seed66 --thinking on --maxNewTokens 8192 --seed 66
python src/runInference.py --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --name full-qwen3-4b-think-seed73 --thinking on --maxNewTokens 8192 --seed 73
python src/runInference.py --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --name full-qwen3-4b-think-seed137 --thinking on --maxNewTokens 8192 --seed 137
python src/runInference.py --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --name full-qwen3-4b-think-seed255 --thinking on --maxNewTokens 8192 --seed 255
python src/runInference.py --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --name full-qwen3-8b-think --thinking on --maxNewTokens 8192 --maxModelLen 10240
python src/runInference.py --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --name full-qwen3-8b-think-seed66 --thinking on --maxNewTokens 8192 --maxModelLen 10240 --seed 66
python src/runInference.py --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --name full-qwen3-8b-think-seed73 --thinking on --maxNewTokens 8192 --maxModelLen 10240 --seed 73
```

In Colab, put `!` in front of each command.

**Things to know:**

- **Runs resume.** If `outputs/runs/<name>.jsonl` already exists, finished questions are skipped and new results are appended. Delete the file to start a run from scratch.
- **Quick test.** Add `--limit 20` to run only the first 20 prompts, but use a different `--name` (for example `test-qwen25coder-1.5b`) so a partial file is not mistaken for a full run.
- **Order.** The Task 3 runs need all five Qwen3-4B thinking seeds (42, 66, 73, 137 and 255).
- **Results.** When the runs finish, copy `outputs/runs/` back to the local computer if inference ran on Colab.

### 3.3 Generation time and expected results on one L4

Generation time excludes model loading, which takes a few minutes. EX is execution accuracy over all 1,034 questions after scoring (step 4).

| Run name | Generation time | Expected EX (%) |
|---|---|---|
| full-qwen25coder-0.5b | 0.3 min | 45.3 |
| full-qwen25coder-1.5b | 1.5 min | 55.8 |
| full-qwen25coder-3b | 0.6 min | 67.9 |
| full-qwen25coder-7b | 1.3 min | 80.0 |
| full-qwen25coder-7b-awq | 0.7 min | 80.4 |
| full-qwen3-1.7b-nothink | 0.6 min | 60.2 |
| full-qwen3-4b-nothink | 0.9 min | 71.2 |
| full-qwen3-8b-nothink | 1.6 min | 74.0 |
| full-qwen3-1.7b-think (seeds 42 / 66 / 73) | 19 to 22 min each | 71.8 / 72.0 / 72.3 |
| full-qwen3-4b-think (seeds 42 / 66 / 73 / 137 / 255) | 35 to 41 min each | 79.4 / 78.7 / 79.0 / 78.9 / 78.7 |
| full-qwen3-8b-think (seeds 42 / 66 / 73) | 88 to 94 min each | 78.9 / 79.1 / 79.9 |

**Fastest check:** the greedy configurations are deterministic, so `full-qwen25coder-1.5b` (about 2 minutes) or `full-qwen25coder-7b` should reproduce their EX exactly. Sampled thinking runs can differ slightly across GPU types and vLLM versions, but should stay within about 1 point.

---

## 4. Scoring (local)

`scoring.py` executes every predicted and gold query on its SQLite database. It computes EM, EX, EX with gold values, component F1, the error bucket and whether each output timed out (30 second limit). It writes `outputs/runs/<name>_scored.jsonl` and `<name>_summary.json`.

Each run takes about 1 to 2 minutes:

```bash
for f in outputs/runs/full-*.jsonl; do
  case "$f" in *_scored.jsonl) continue;; esac
  python src/scoring.py --run "$f"
done
```

---

## 5. Task 3: execution-based self-consistency (local)

```bash
python src/selfConsistency.py
python src/scoring.py --run outputs/runs/full-qwen3-4b-think-sc3.jsonl
python src/scoring.py --run outputs/runs/full-qwen3-4b-think-sc5.jsonl
```

`selfConsistency.py` takes about 30 seconds. It votes over the five scored Qwen3-4B thinking seeds using the rules in the report:

1. Group the candidate queries by identical result rows, ignoring row order.
2. Drop candidates that crashed, timed out or produced no SQL.
3. The largest group wins.
4. Ties go to the group holding the shortest reasoning trace.
5. If every candidate failed, seed 42's query stands.

It writes the voted runs `full-qwen3-4b-think-sc3` (seeds 42, 66, 73) and `full-qwen3-4b-think-sc5` (all five), and puts its analysis in `outputs/selfConsistency/`. **Score the five seed runs (step 4) before running it.** The two new runs are then scored like any other.

---

## 6. Statistics, trace analysis and figures (local)

```bash
python src/summarizeRuns.py     # Task 1 and Task 3 tables and tests -> outputs/summary/        (about 10 s)
python src/analyzeTraces.py     # Task 2 features, tests and figures -> outputs/traceAnalysis/  (about 6 s)
python src/plotResults.py       # Task 1 and Task 3 figures -> outputs/figures/                 (about 1 s)
```

- `summarizeRuns.py` covers accuracy by difficulty with Wilson intervals, the error buckets, component F1, efficiency, McNemar tests with Holm correction and clustered bootstrap intervals, seed means, the return on thinking, EM parser failures and pessimistic EX. Each table is a CSV and all of them are also in `outputs/summary/summary.md`.
- `analyzeTraces.py` uses Qwen3-4B thinking seed 42 as the primary run and the other 10 thinking runs as replications.
- Runs that are missing or not yet scored are skipped with a message, so these scripts also work on a partial set of runs.

**Where each report table and figure comes from:**

| Report | Output |
|---|---|
| Tables 3, 4, 5, 6, 7, 8 | `outputs/summary/accuracy.csv`, `mcnemar.csv` and `mcnemar_pessimistic.csv`, `efficiency.csv`, `thinking.csv`, `em_parser.csv`, `errors.csv` and `coincidence.csv` |
| Tables 9 to 12 | `outputs/traceAnalysis/descriptives.csv`, `tests.csv` and `auroc.csv`, `regression.csv`, `replication.csv` |
| Table 13 | `outputs/selfConsistency/by_difficulty.csv` and `outputs/summary/mcnemar.csv` |
| Figures 1, 2, 3, 6, 7 | `outputs/figures/size_curves.png`, `accuracy_by_difficulty.png`, `cost_tradeoff.png`, `compute_ladder.png`, `cost_tradeoff_voting.png` |
| Figures 4, 5 | `outputs/traceAnalysis/boxplots_full-qwen3-4b-think.png`, `accuracy_by_length_full-qwen3-4b-think.png` |
| Appendices B, C | `outputs/summary/settings.csv`, `components.csv` |

---

## 7. Full order at a glance

```
local    2.1 to 2.3   environment, Spider, evaluator
local    2.4          prepareSubset.py --n 1034, buildPrompts.py, sanityCheck.py
GPU      3            runInference.py for the 19 runs
local    4            scoring.py on every run
local    5            selfConsistency.py, then score sc3 and sc5
local    6            summarizeRuns.py, analyzeTraces.py, plotResults.py
```

## Folder layout

```
src/
  spiderUtils.py       shared paths, seed 42, schema building and SQL extraction (standard library only)
  prepareSubset.py     difficulty labels and the evaluation set (the whole dev split with --n 1034)
  buildPrompts.py      zero-shot prompts with CREATE TABLE schemas
  sanityCheck.py       extraction and scorer checks against the official evaluator
  runInference.py      vLLM generation for one configuration (GPU)
  scoring.py           EM, EX, EX with gold values, component F1, error buckets
  selfConsistency.py   Task 3 voting
  summarizeRuns.py     Task 1 and Task 3 tables and statistics
  analyzeTraces.py     Task 2 trace features and statistics
  plotResults.py       figures
data/  eval/  outputs/ created by the steps above (not included in the repository)
```
