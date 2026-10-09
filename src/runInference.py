"""
how to run on colab
    !python src/runInference.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --name qwen25coder-1.5b --thinking none
    !python src/runInference.py --model Qwen/Qwen3-4B --name qwen3-4b-think --thinking on --maxNewTokens 8192
    !python src/runInference.py --model Qwen/Qwen3-4B --name qwen3-4b-think-seed2 --thinking on --maxNewTokens 8192 --seed 2
    !python src/runInference.py --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --name qwen25coder-7b-awq --thinking none --dtype auto

scoring on local comp
    python src/scoring.py --run outputs/runs/<run name>.jsonl
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone

from spiderUtils import extractSql, outDir, randomSeed, readJsonl


# decoding settings per thinking mode as temperature then top p then top k
# temperature 0 means greedy and top k of -1 means vllm doesnt use top k at all
# the on settings are what the qwen3 model card recommends for thinking mode
decodingByMode = {
    "none": (0.0, 1.0, -1),
    "off": (0.0, 1.0, -1),
    "on": (0.6, 0.95, 20),
}

# how much gpu memory vllm is allowed to grab
gpuMemoryShare = 0.90

# the fields copied from each prompt row into its output row
promptFieldsToKeep = ["dev_idx", "db_id", "difficulty", "question", "gold"]


def splitReasoning(tokenIds, thinkEndId: int) -> tuple:
    """
    splits generated tokens at the last </think> token into reasoning and answer
    splitting on the token id instead of the text works even if decoding hides the think tags
    returns reasoning ids then answer ids then whether thinking actually finished
    not finished means it hit the token limit while still reasoning
    """
    tokenIds = list(tokenIds)

    # walk backwards so we land on the last closing tag if theres more than one
    lastThinkEndPosition = None
    for position in range(len(tokenIds) - 1, -1, -1):
        if tokenIds[position] == thinkEndId:
            lastThinkEndPosition = position
            break

    if lastThinkEndPosition is None:
        return tokenIds, [], False

    reasoningIds = tokenIds[:lastThinkEndPosition]
    answerIds = tokenIds[lastThinkEndPosition + 1:]

    return reasoningIds, answerIds, True


def findQuantization(modelId: str, revision: str):
    """
    reads the quantization straight from the models own config so the run log cant claim the wrong thing
    full precision models have no quantization_config at all so those come back as none
    """
    from transformers import AutoConfig

    try:
        modelConfig = AutoConfig.from_pretrained(modelId, revision = revision)
    except Exception as error:
        # better to log unknown than to guess
        print(f"couldnt read the model config so quantization is logged as unknown because {error}")
        return "unknown"

    quantSettings = getattr(modelConfig, "quantization_config", None)

    if quantSettings is None:
        return "none"

    # transformers sometimes keeps this as a plain dict and sometimes as an object
    if not isinstance(quantSettings, dict):
        quantSettings = quantSettings.to_dict()

    return {"method": quantSettings.get("quant_method"), "bits": quantSettings.get("bits")}


def findPromptsToRun(promptsPath, outputPath, limit) -> tuple:
    allPrompts = readJsonl(promptsPath)

    # limit is for pilots and None means run everything
    if limit is not None:
        allPrompts = allPrompts[:limit]

    # anything already in the output file got finished in an earlier session
    doneIds = set()
    if outputPath.exists():
        for row in readJsonl(outputPath):
            doneIds.add(row["dev_idx"])

    promptsToRun = []
    for prompt in allPrompts:
        if prompt["dev_idx"] not in doneIds:
            promptsToRun.append(prompt)

    return allPrompts, doneIds, promptsToRun


def buildOutputRow(prompt: dict, generation, args, secondsPerExample: float, tokenizer, thinkEndId) -> dict:
    # output keys go into the run files that scoring and task 2 read so they keep their underscores
    row = {}
    for field in promptFieldsToKeep:
        row[field] = prompt[field]

    row["model"] = args.model
    row["thinking"] = args.thinking
    row["raw_output"] = generation.text
    row["finish_reason"] = generation.finish_reason
    row["n_output_tokens"] = len(generation.token_ids)
    row["amortized_sec"] = secondsPerExample

    if args.thinking == "on":
        reasoningIds, answerIds, thinkClosed = splitReasoning(generation.token_ids, thinkEndId)

        # saved separately because task 2 analyzes the reasoning on its own
        row["reasoning_text"] = tokenizer.decode(reasoningIds, skip_special_tokens = True)
        row["answer_text"] = tokenizer.decode(answerIds, skip_special_tokens = True)
        row["n_reasoning_tokens"] = len(reasoningIds)
        row["think_closed"] = thinkClosed

        if thinkClosed:
            sql, status = extractSql(row["answer_text"])
        else:
            sql = None
            status = "truncated_in_reasoning"
    else:
        sql, status = extractSql(generation.text)

    row["pred_sql"] = sql
    row["extract_status"] = status

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, help = "hugging face model id")
    parser.add_argument("--name", required = True, help = "run name which becomes the output file name")
    parser.add_argument("--thinking", choices = ["none", "on", "off"], required = True)
    parser.add_argument("--revision", default = None, help = "model commit hash and leaving it out means latest")
    parser.add_argument("--prompts", default = str(outDir / "prompts.jsonl"))
    parser.add_argument("--limit", type = int, default = None, help = "only the first n prompts for pilots")
    parser.add_argument("--maxNewTokens", type = int, default = 512)
    parser.add_argument("--maxModelLen", type = int, default = 16384)
    parser.add_argument("--chunk", type = int, default = 100, help = "prompts per vllm batch and per save")
    parser.add_argument("--seed", type = int, default = randomSeed, help = "sampling seed which only matters for thinking runs since the rest are greedy")
    parser.add_argument(
        "--dtype",
        default = None,
        choices = ["bfloat16", "float16", "auto"],
        help = "leave out for bfloat16 when the gpu supports it and float16 otherwise and use auto to let vllm decide",
    )
    args = parser.parse_args()

    # imported in here so the rest of src still imports fine on the mac where vllm cant install
    import torch
    import transformers
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams

    runsDir = outDir / "runs"
    runsDir.mkdir(parents = True, exist_ok = True)
    outputPath = runsDir / f"{args.name}.jsonl"
    configPath = runsDir / f"{args.name}_config.json"

    allPrompts, doneIds, promptsToRun = findPromptsToRun(args.prompts, outputPath, args.limit)
    print(f"{len(allPrompts)} prompts with {len(doneIds)} already done and {len(promptsToRun)} to run")

    if len(promptsToRun) == 0:
        return

    # pin the exact checkpoint so the report can name it and a rerun gets the same weights
    revision = HfApi().model_info(args.model, revision = args.revision).sha

    # quantized checkpoints can need a specific precision so --dtype lets vllm pick for those
    # otherwise bfloat16 and the t4 cant do bfloat16 so it falls back to float16 there
    if args.dtype is not None:
        dtype = args.dtype
    elif torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    else:
        dtype = "float16"

    quantization = findQuantization(args.model, revision)

    llm = LLM(
        model = args.model,
        revision = revision,
        dtype = dtype,
        seed = args.seed,
        max_model_len = args.maxModelLen,
        gpu_memory_utilization = gpuMemoryShare,
    )
    tokenizer = llm.get_tokenizer()

    # what vllm actually ran with since auto or a quantized model can end up different from what was asked for
    try:
        dtypeUsed = str(llm.llm_engine.model_config.dtype)
    except Exception:
        dtypeUsed = "unknown"

    # qwen3 needs to be told whether to think and models without a thinking mode get nothing extra
    templateOptions = {}
    if args.thinking != "none":
        templateOptions["enable_thinking"] = (args.thinking == "on")

    promptTexts = []
    for prompt in promptsToRun:
        promptText = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize = False,
            add_generation_prompt = True,
            **templateOptions,
        )
        promptTexts.append(promptText)

    temperature, topP, topK = decodingByMode[args.thinking]
    samplingSettings = SamplingParams(
        temperature = temperature,
        top_p = topP,
        top_k = topK,
        max_tokens = args.maxNewTokens,
        seed = args.seed,
    )

    thinkEndId = None
    if args.thinking == "on":
        thinkEndId = tokenizer.convert_tokens_to_ids("</think>")

    sessionStart = time.perf_counter()

    for chunkStart in range(0, len(promptsToRun), args.chunk):
        chunkEnd = chunkStart + args.chunk
        chunkPrompts = promptsToRun[chunkStart:chunkEnd]
        chunkTexts = promptTexts[chunkStart:chunkEnd]

        # vllm runs the whole chunk together so per example time is the chunk time split evenly
        chunkStartTime = time.perf_counter()
        results = llm.generate(chunkTexts, samplingSettings)
        secondsPerExample = (time.perf_counter() - chunkStartTime) / len(chunkPrompts)

        # results come back in the same order as the prompts went in
        with open(outputPath, "a") as outputFile:
            for prompt, result in zip(chunkPrompts, results):
                generation = result.outputs[0]
                row = buildOutputRow(prompt, generation, args, secondsPerExample, tokenizer, thinkEndId)
                outputFile.write(json.dumps(row) + "\n")

        chunkNumber = chunkStart // args.chunk + 1
        print(f"chunk {chunkNumber} done with {len(chunkPrompts)} examples at {secondsPerExample:.2f} s per example amortized")

    # every session gets logged so a resumed run shows all of its pieces
    # keys stay with underscores since this file is data
    if configPath.exists():
        runConfig = json.loads(configPath.read_text())
    else:
        runConfig = {"sessions": []}

    runConfig["model"] = args.model
    runConfig["revision"] = revision
    runConfig["thinking"] = args.thinking
    runConfig["dtype"] = dtype
    runConfig["dtype_used"] = dtypeUsed
    runConfig["quantization"] = quantization
    runConfig["decoding"] = {
        "temperature": temperature,
        "top_p": topP,
        "top_k": topK,
        "max_new_tokens": args.maxNewTokens,
        "seed": args.seed,
    }
    runConfig["max_model_len"] = args.maxModelLen
    runConfig["chunk"] = args.chunk

    sessionLog = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec = "seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "n_examples": len(promptsToRun),
        "wall_sec": round(time.perf_counter() - sessionStart, 1),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    runConfig["sessions"].append(sessionLog)

    configPath.write_text(json.dumps(runConfig, indent = 2))
    print(f"done with outputs in {outputPath} and config in {configPath}")


if __name__ == "__main__":
    main()