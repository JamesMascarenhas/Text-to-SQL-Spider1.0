"""
builds the prompt for every example in the evaluation subset
writes outputs/prompts.jsonl which holds chat messages that are the same for every model
that file plus src is all colab needs so the spider data never has to leave the mac
each models tokenizer turns the messages into its own prompt format later at inference time
runs on the mac after prepareSubset.py

how to run
    python src/buildPrompts.py              schema only as CREATE TABLE statements
    python src/buildPrompts.py --nRows 3    also 3 example rows per table
"""

import argparse
import json
import statistics

from spiderUtils import buildMessages, buildSchemaText, outDir, readJsonl, writeJsonl


def buildSchemaTextsForSubset(subsetRows: list, nRows: int) -> dict:
    # only 20 databases in dev so build each schema once instead of once per question
    dbIds = set()
    for row in subsetRows:
        dbIds.add(row["db_id"])

    schemaTextByDb = {}
    for dbId in dbIds:
        schemaTextByDb[dbId] = buildSchemaText(dbId, nRows = nRows)

    return schemaTextByDb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nRows", type = int, default = 0, help = "example rows per table in the schema")
    args = parser.parse_args()

    subsetRows = readJsonl(outDir / "subset.jsonl")
    schemaTextByDb = buildSchemaTextsForSubset(subsetRows, args.nRows)

    # each prompt row is the subset row plus its messages so colab has everything in one place
    promptRows = []
    for subsetRow in subsetRows:
        promptRow = dict(subsetRow)
        promptRow["messages"] = buildMessages(subsetRow["question"], schemaTextByDb[subsetRow["db_id"]])
        promptRows.append(promptRow)

    writeJsonl(promptRows, outDir / "prompts.jsonl")

    # records the schema choice so the methodology section can quote it straight from here
    # keys stay with underscores since this is a data file
    promptConfig = {
        "n_examples": len(promptRows),
        "schema_format": "CREATE TABLE statements from sqlite_master",
        "n_rows": args.nRows,
    }

    with open(outDir / "prompts_config.json", "w") as configFile:
        json.dump(promptConfig, configFile, indent = 2)

    # prompt length in characters to keep an eye on how big the biggest schema gets
    promptLengths = []
    for promptRow in promptRows:
        totalLength = 0
        for message in promptRow["messages"]:
            totalLength = totalLength + len(message["content"])
        promptLengths.append(totalLength)

    print(f"wrote {len(promptRows)} prompts to {outDir / 'prompts.jsonl'}")
    print(f"prompt length in characters with median {statistics.median(promptLengths):.0f} and max {max(promptLengths)}")

    # print one so the schema can be eyeballed before uploading anything
    print("\n--- example prompt user message ---\n" + promptRows[0]["messages"][1]["content"])


if __name__ == "__main__":
    main()