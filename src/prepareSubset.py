"""
labels every spider dev example with its official difficulty and then draws the evaluation subset
the sample is stratified and proportional so it keeps the same mix of easy medium hard and extra as the full dev set
that way overall accuracy on the subset is a fair estimate for full dev and every difficulty level is guaranteed to show up
an examples id is just its position in dev.json counting from 0 since dev.json has no id field
runs on the mac

how to run
    python src/prepareSubset.py --n 400      stratified subset of 400
    python src/prepareSubset.py --n 1034     the whole dev set
"""

import argparse
import json
import random

from scoring import getDifficulty
from spiderUtils import dataDir, difficultyLevels, loadJson, outDir, randomSeed, writeJsonl


def splitProportionally(countsByLevel: dict, nTotal: int) -> dict:
    """
    splits nTotal across difficulty levels in proportion to how big each level is
    rounding down first and then handing the leftover spots to the levels that lost the most from rounding
    that way the pieces always add up to exactly nTotal
    """
    grandTotal = sum(countsByLevel.values())

    exactShares = {}
    for level, count in countsByLevel.items():
        exactShares[level] = nTotal * count / grandTotal

    allocation = {}
    for level, exactShare in exactShares.items():
        allocation[level] = int(exactShare)

    leftover = nTotal - sum(allocation.values())

    def roundingLoss(level):
        return exactShares[level] - allocation[level]

    # biggest rounding losses get the leftover spots first
    levelsByLoss = sorted(exactShares, key = roundingLoss, reverse = True)

    for level in levelsByLoss[:leftover]:
        allocation[level] = allocation[level] + 1

    return allocation


def labelDevSet() -> list:
    devExamples = loadJson(dataDir / "dev.json")

    labeledRows = []

    for devIdx, example in enumerate(devExamples):
        # the keys go into every downstream file so they keep their underscores
        row = {
            "dev_idx": devIdx,
            "db_id": example["db_id"],
            "question": example["question"],
            "gold": example["query"],
            "difficulty": getDifficulty(example["query"], example["db_id"]),
        }
        labeledRows.append(row)

    return labeledRows


def getDevIdx(row: dict) -> int:
    return row["dev_idx"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type = int, default = 400)
    parser.add_argument("--seed", type = int, default = randomSeed)
    args = parser.parse_args()

    # label everything first and save it since task 2 and 3 might want the full labeled set
    labeledRows = labelDevSet()
    writeJsonl(labeledRows, outDir / "dev_all_labeled.jsonl")

    # how many examples sit in each level
    # built in the order levels first show up in dev.json which also decides ties when handing out leftovers
    countsByLevel = {}
    for row in labeledRows:
        level = row["difficulty"]
        if level not in countsByLevel:
            countsByLevel[level] = 0
        countsByLevel[level] = countsByLevel[level] + 1

    # asking for more than exists just means take all of dev
    nSubset = min(args.n, len(labeledRows))
    allocation = splitProportionally(countsByLevel, nSubset)

    rowsByLevel = {}
    for row in labeledRows:
        level = row["difficulty"]
        if level not in rowsByLevel:
            rowsByLevel[level] = []
        rowsByLevel[level].append(row)

    # one seeded generator drawing the levels in a fixed order so the same seed always gives the same subset
    randomGenerator = random.Random(args.seed)

    subsetRows = []
    for level in difficultyLevels:
        drawnRows = randomGenerator.sample(rowsByLevel[level], allocation[level])
        subsetRows = subsetRows + drawnRows

    # back in dev.json order so the files are easy to scan and line up with the original data
    subsetRows.sort(key = getDevIdx)

    writeJsonl(subsetRows, outDir / "subset.jsonl")

    # the assignment wants the exact ids recorded so this file is the record of which examples were used
    subsetIds = []
    for row in subsetRows:
        subsetIds.append(row["dev_idx"])

    with open(outDir / "subset_ids.json", "w") as idsFile:
        json.dump({"n": nSubset, "seed": args.seed, "dev_idx": subsetIds}, idsFile)

    # quick summary to eyeball that the mix came out right
    print(f"{'level':8}{'dev':>6}{'subset':>8}")
    for level in difficultyLevels:
        print(f"{level:8}{countsByLevel[level]:>6}{allocation[level]:>8}")
    print(f"{'total':8}{len(labeledRows):>6}{nSubset:>8}")

    subsetDbIds = set()
    for row in subsetRows:
        subsetDbIds.add(row["db_id"])

    print(f"databases in subset {len(subsetDbIds)}")
    print(f"wrote {outDir / 'subset.jsonl'} and subset_ids.json")


if __name__ == "__main__":
    main()