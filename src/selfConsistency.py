"""
task 3 which is execution based self consistency for qwen3 4b with thinking on
every seed of the thinking run is one independent sample of reasoning for every question
each sampled query gets run on the database and queries that return the same rows count as the same answer
the answer with the most votes wins
the gold answer is never looked at while voting so this is a fair inference time method
runs on the mac after the seed runs are scored

voting rules fixed before seeing any self consistency results
  candidates     the query from each seed
  same answer    same rows returned ignoring row order
  cant win       candidates that crash or time out or have no sql
  winner         the biggest group
  ties           the tied group holding the shortest reasoning trace wins
                 following hassid et al 2025 who found shorter chains for the same question are more often right
                 and matching task 2 where longer traces went with wrong answers
  chosen query   the shortest trace inside the winning group which only matters for em since the rows are identical
  all failed     seed 42s query stands since its the original baseline
  sensitivity    the same vote with ties going to the earliest seed instead gets reported too
                 along with how many questions a tie actually decided

a quirk of the official scorer worth knowing
  ex strips every DISTINCT before running queries including the one inside COUNT(DISTINCT x)
  the vote runs queries as written so two queries in the same group can still score differently on ex
  that means picking which query to use inside a group can flip ex even though the answer didnt change
  so the main table also counts fixes and breaks that came only from that choice
  and a keep baseline row shows the vote when seed 42s query is kept whenever its in the winning group
  under that rule ex only changes when the vote really switches to a different answer

what it writes
  outputs/runs/<name>-sc<N>.jsonl plus a config so scoring.py can score it like any other run
  outputs/selfConsistency/ with the vote details and a selfConsistency.md holding
    main result     baseline vs voted vs the upper bound where at least one candidate is right
                    with the gain and a 95 percent interval from resampling gold queries
    by difficulty   the same split by difficulty
    scaling         voted accuracy for every subset size averaged over every subset of seeds
    agreement       accuracy by how many candidates agreed which is the confidence signal
    cost            tokens and generation time for N samples vs one
    examples        questions the vote fixed and questions it broke

how to run
    python src/selfConsistency.py
    python src/selfConsistency.py --base full-qwen3-4b-think --seeds 42 66 73 137 255 --writeN 3 5
then score the new run files as usual
"""

import argparse
import copy
import itertools
import json
import sqlite3
import statistics
import time

import pandas as pd

# importing scoring puts the eval repo on the path and brings the shared time limit with it
from scoring import decodeIgnoringBadBytes, predTimeLimitSeconds
from exec_eval import postprocess, replace_cur_year  # noqa: E402
from spiderUtils import difficultyLevels, getDbPath, outDir, readJsonl, writeJsonl  # noqa: E402
from summarizeRuns import exactMcNemar, goldGroupKey, pairedGainInterval, wilsonInterval  # noqa: E402


runsDir = outDir / "runs"
consistencyDir = outDir / "selfConsistency"

defaultBase = "full-qwen3-4b-think"
defaultSeeds = [42, 66, 73, 137, 255]

# fields copied from the chosen candidate into the voted run file
# scores are left out on purpose since scoring.py adds fresh ones
runFields = [
    "dev_idx", "db_id", "difficulty", "question", "gold", "model", "thinking",
    "raw_output", "finish_reason", "reasoning_text", "answer_text", "think_closed",
    "pred_sql", "extract_status",
]


# loading

def runNameForSeed(baseName: str, seed: int) -> str:
    # seed 42 is the original run and has no suffix
    if seed == 42:
        return baseName
    return f"{baseName}-seed{seed}"


def loadCandidateRuns(baseName: str, seeds: list) -> tuple:
    runsBySeed = {}
    usedSeeds = []

    for seed in seeds:
        scoredPath = runsDir / f"{runNameForSeed(baseName, seed)}_scored.jsonl"

        if not scoredPath.exists():
            print(f"seed {seed} isnt scored yet so its left out")
            continue

        rowsById = {}
        for row in readJsonl(scoredPath):
            rowsById[row["dev_idx"]] = row

        runsBySeed[seed] = rowsById
        usedSeeds.append(seed)

    return runsBySeed, usedSeeds


# executing candidates

def executeForVote(dbId: str, predSql) -> object:
    """
    runs one candidate and returns a signature of its rows or None if it cant take part in the vote
    the signature is the sorted list of rows as text so row order doesnt matter but duplicates still do
    same cleanup and same time limit as the scorer so a query that hangs there hangs here too
    """
    if predSql is None or predSql == "":
        return None

    query = replace_cur_year(postprocess(predSql))

    connection = sqlite3.connect(str(getDbPath(dbId)))
    connection.text_factory = decodeIgnoringBadBytes

    deadline = time.monotonic() + predTimeLimitSeconds

    def pastDeadline():
        if time.monotonic() > deadline:
            return 1
        return 0

    connection.set_progress_handler(pastDeadline, 10000)

    try:
        cursor = connection.cursor()
        cursor.execute(query)
        resultRows = cursor.fetchall()
    except Exception:
        # crashes and timeouts both just drop out of the vote
        return None
    finally:
        connection.close()

    rowTexts = []
    for resultRow in resultRows:
        rowTexts.append(repr(resultRow))
    rowTexts.sort()

    return tuple(rowTexts)


def buildCandidateTable(runsBySeed: dict, usedSeeds: list) -> dict:
    # every candidate gets executed once up front so any subset of seeds can be voted on without rerunning anything
    firstSeed = usedSeeds[0]
    questionIds = sorted(runsBySeed[firstSeed].keys())

    candidatesById = {}

    for position, devIdx in enumerate(questionIds):
        candidates = []

        for seed in usedSeeds:
            candidateRow = runsBySeed[seed][devIdx]
            signature = executeForVote(candidateRow["db_id"], candidateRow["pred_sql"])

            candidates.append({
                "seed": seed,
                "row": candidateRow,
                "signature": signature,
                "ex": candidateRow["ex"],
                "tokens": candidateRow["n_reasoning_tokens"],
            })

        candidatesById[devIdx] = candidates

        if (position + 1) % 100 == 0:
            print(f"  executed candidates for {position + 1} of {len(questionIds)} questions")

    return candidatesById


# voting

def shortestPosition(candidates: list, positions: list) -> int:
    # the candidate with the fewest reasoning tokens and the earlier seed if two are exactly equal
    bestPosition = positions[0]
    for position in positions[1:]:
        if candidates[position]["tokens"] < candidates[bestPosition]["tokens"]:
            bestPosition = position
    return bestPosition


def vote(candidates: list, tieBreak: str = "shortest") -> dict:
    """
    candidates come in the fixed seed order and the first one is the baseline
    tieBreak is shortest for the main rule or seedOrder for the sensitivity check
    returns which candidate wins plus how much agreement there was and whether a tie decided it
    """
    # each distinct result gets the positions of the candidates that produced it
    groups = {}
    for position, candidate in enumerate(candidates):
        if candidate["signature"] is None:
            continue
        if candidate["signature"] not in groups:
            groups[candidate["signature"]] = []
        groups[candidate["signature"]].append(position)

    # nothing ran so the baseline stands
    if len(groups) == 0:
        return {"chosenPosition": 0, "winnerVotes": 0, "nGroups": 0, "nValid": 0, "decidedByTie": False}

    biggestSize = 0
    for positions in groups.values():
        biggestSize = max(biggestSize, len(positions))

    tiedGroups = []
    for positions in groups.values():
        if len(positions) == biggestSize:
            tiedGroups.append(positions)

    # pick among the tied groups
    winnerPositions = tiedGroups[0]
    for positions in tiedGroups[1:]:
        if tieBreak == "shortest":
            challenger = candidates[shortestPosition(candidates, positions)]["tokens"]
            holder = candidates[shortestPosition(candidates, winnerPositions)]["tokens"]
            if challenger < holder:
                winnerPositions = positions
        else:
            if positions[0] < winnerPositions[0]:
                winnerPositions = positions

    if tieBreak == "shortest":
        chosenPosition = shortestPosition(candidates, winnerPositions)
    else:
        chosenPosition = winnerPositions[0]

    nValid = 0
    for positions in groups.values():
        nValid = nValid + len(positions)

    return {
        "chosenPosition": chosenPosition,
        "winnerVotes": biggestSize,
        "nGroups": len(groups),
        "nValid": nValid,
        "decidedByTie": len(tiedGroups) > 1,
    }


def voteOverSeeds(candidatesById: dict, seedSubset: list, tieBreak: str = "shortest") -> dict:
    # votes using only the given seeds while keeping the fixed order
    resultsById = {}

    for devIdx, candidates in candidatesById.items():
        subsetCandidates = []
        for candidate in candidates:
            if candidate["seed"] in seedSubset:
                subsetCandidates.append(candidate)

        voteResult = vote(subsetCandidates, tieBreak)
        chosen = subsetCandidates[voteResult["chosenPosition"]]
        baseline = subsetCandidates[0]

        # same rows as the baseline means the vote kept the baseline answer and only the query text differs
        sameRowsAsBaseline = baseline["signature"] is not None and chosen["signature"] == baseline["signature"]

        if sameRowsAsBaseline:
            votedExKeepBaseline = baseline["ex"]
        else:
            votedExKeepBaseline = chosen["ex"]

        anyRight = 0
        for candidate in subsetCandidates:
            if candidate["ex"] == 1:
                anyRight = 1

        resultsById[devIdx] = {
            "chosenSeed": chosen["seed"],
            "votedEx": chosen["ex"],
            "baselineEx": subsetCandidates[0]["ex"],
            "anyRight": anyRight,
            "winnerVotes": voteResult["winnerVotes"],
            "nGroups": voteResult["nGroups"],
            "nValid": voteResult["nValid"],
            "decidedByTie": voteResult["decidedByTie"],
            "sameRowsAsBaseline": sameRowsAsBaseline,
            "votedExKeepBaseline": votedExKeepBaseline,
            "n": len(subsetCandidates),
        }

    return resultsById


# run file for scoring

def writeVotedRun(baseName: str, candidatesById: dict, seedSubset: list, runsBySeed: dict) -> str:
    resultsById = voteOverSeeds(candidatesById, seedSubset)
    votedName = f"{baseName}-sc{len(seedSubset)}"

    votedRows = []
    for devIdx, candidates in candidatesById.items():
        result = resultsById[devIdx]

        chosenRow = None
        totalSeconds = 0.0
        totalOutputTokens = 0
        totalReasoningTokens = 0
        for candidate in candidates:
            if candidate["seed"] not in seedSubset:
                continue
            # the cost of self consistency is every sample not just the winner so these get summed
            totalSeconds = totalSeconds + candidate["row"]["amortized_sec"]
            totalOutputTokens = totalOutputTokens + candidate["row"]["n_output_tokens"]
            totalReasoningTokens = totalReasoningTokens + candidate["row"]["n_reasoning_tokens"]
            if candidate["seed"] == result["chosenSeed"]:
                chosenRow = candidate["row"]

        votedRow = {}
        for field in runFields:
            votedRow[field] = chosenRow[field]

        votedRow["amortized_sec"] = totalSeconds
        votedRow["n_output_tokens"] = totalOutputTokens
        votedRow["n_reasoning_tokens"] = totalReasoningTokens
        votedRow["sc_n"] = result["n"]
        votedRow["sc_chosen_seed"] = result["chosenSeed"]
        votedRow["sc_winner_votes"] = result["winnerVotes"]
        votedRow["sc_groups"] = result["nGroups"]
        votedRow["sc_valid"] = result["nValid"]
        votedRows.append(votedRow)

    writeJsonl(votedRows, runsDir / f"{votedName}.jsonl")

    # the config is the baseline config plus a record of the strategy so the summary can label it
    with open(runsDir / f"{baseName}_config.json") as configFile:
        votedConfig = json.load(configFile)

    votedConfig = copy.deepcopy(votedConfig)
    votedConfig["strategy"] = {
        "name": "execution based self consistency",
        "n_samples": len(seedSubset),
        "seeds": seedSubset,
        "same_answer": "identical rows ignoring row order",
        "excluded": "candidates that crash or time out or have no sql",
        "tie_break": "tied group holding the shortest reasoning trace following hassid et al 2025",
        "chosen_query": "shortest reasoning trace in the winning group",
        "all_failed": "baseline seed 42 query",
        "cost_fields": "amortized_sec and token counts are summed over all samples",
    }

    with open(runsDir / f"{votedName}_config.json", "w") as configFile:
        json.dump(votedConfig, configFile, indent = 2)

    return votedName


# analysis tables

def mainTable(resultsById: dict, groupKeyById: dict, exKey: str = "votedEx") -> dict:
    # exKey picks which version of the vote gets summarized so the same table works for the sensitivity rows
    results = list(resultsById.values())
    n = len(results)

    # the gain over the baseline with an interval that keeps paraphrase pairs together
    groupKeys = []
    baselineValues = []
    votedValues = []
    for devIdx, result in resultsById.items():
        groupKeys.append(groupKeyById[devIdx])
        baselineValues.append(result["baselineEx"])
        votedValues.append(result[exKey])
    gainLow, gainHigh = pairedGainInterval(groupKeys, baselineValues, votedValues)

    baselineRight = 0
    votedRight = 0
    anyRight = 0
    fixedCount = 0
    brokenCount = 0
    fixedByGroupChoice = 0
    brokenByGroupChoice = 0
    for result in results:
        baselineRight = baselineRight + result["baselineEx"]
        votedRight = votedRight + result[exKey]
        anyRight = anyRight + result["anyRight"]

        isFixed = result["baselineEx"] == 0 and result[exKey] == 1
        isBroken = result["baselineEx"] == 1 and result[exKey] == 0

        if isFixed:
            fixedCount = fixedCount + 1
        if isBroken:
            brokenCount = brokenCount + 1

        # the answer stayed the same and only the chosen query text changed the score
        if isFixed and result["sameRowsAsBaseline"]:
            fixedByGroupChoice = fixedByGroupChoice + 1
        if isBroken and result["sameRowsAsBaseline"]:
            brokenByGroupChoice = brokenByGroupChoice + 1

    votedLow, votedHigh = wilsonInterval(votedRight, n)

    return {
        "n_questions": n,
        "baseline_ex": baselineRight / n,
        "voted_ex": votedRight / n,
        "voted_ci_low": votedLow,
        "voted_ci_high": votedHigh,
        "gain": (votedRight - baselineRight) / n,
        "gain_ci_low": gainLow,
        "gain_ci_high": gainHigh,
        "upper_bound_any_right": anyRight / n,
        "fixed": fixedCount,
        "broken": brokenCount,
        "fixed_by_group_choice_only": fixedByGroupChoice,
        "broken_by_group_choice_only": brokenByGroupChoice,
        "mcnemar_p": exactMcNemar(brokenCount, fixedCount),
    }


def difficultyTable(resultsById: dict, candidatesById: dict) -> pd.DataFrame:
    tableRows = []

    for level in difficultyLevels + ["all"]:
        levelResults = []
        for devIdx, result in resultsById.items():
            levelName = candidatesById[devIdx][0]["row"]["difficulty"]
            if level == "all" or levelName == level:
                levelResults.append(result)

        n = len(levelResults)
        baselineTotal = 0
        votedTotal = 0
        anyTotal = 0
        for result in levelResults:
            baselineTotal = baselineTotal + result["baselineEx"]
            votedTotal = votedTotal + result["votedEx"]
            anyTotal = anyTotal + result["anyRight"]

        tableRows.append({
            "difficulty": level,
            "n": n,
            "baseline_ex": baselineTotal / n,
            "voted_ex": votedTotal / n,
            "gain": (votedTotal - baselineTotal) / n,
            "upper_bound_any_right": anyTotal / n,
        })

    return pd.DataFrame(tableRows)


def scalingTable(candidatesById: dict, usedSeeds: list) -> pd.DataFrame:
    """
    voted accuracy for every number of samples from 1 up to all of them
    averaged over every subset of that size so no single lucky seed decides the curve
    n of 1 is just the average single run
    """
    tableRows = []

    for subsetSize in range(1, len(usedSeeds) + 1):
        votedScores = []
        keepBaselineScores = []
        upperBounds = []

        for seedSubset in itertools.combinations(usedSeeds, subsetSize):
            resultsById = voteOverSeeds(candidatesById, list(seedSubset))

            votedTotal = 0
            keepBaselineTotal = 0
            anyTotal = 0
            for result in resultsById.values():
                votedTotal = votedTotal + result["votedEx"]
                keepBaselineTotal = keepBaselineTotal + result["votedExKeepBaseline"]
                anyTotal = anyTotal + result["anyRight"]

            votedScores.append(votedTotal / len(resultsById))
            keepBaselineScores.append(keepBaselineTotal / len(resultsById))
            upperBounds.append(anyTotal / len(resultsById))

        tableRows.append({
            "n_samples": subsetSize,
            "n_subsets": len(votedScores),
            "voted_ex_mean": statistics.mean(votedScores),
            "voted_ex_min": min(votedScores),
            "voted_ex_max": max(votedScores),
            "keep_baseline_ex_mean": statistics.mean(keepBaselineScores),
            "upper_bound_mean": statistics.mean(upperBounds),
        })

    return pd.DataFrame(tableRows)


def agreementTable(resultsById: dict) -> tuple:
    # how often the voted answer is right given how many of the samples agreed on it
    rowsByVotes = {}
    for result in resultsById.values():
        votes = result["winnerVotes"]
        if votes not in rowsByVotes:
            rowsByVotes[votes] = []
        rowsByVotes[votes].append(result)

    tableRows = []
    for votes in sorted(rowsByVotes):
        results = rowsByVotes[votes]

        rightTotal = 0
        for result in results:
            rightTotal = rightTotal + result["votedEx"]

        tableRows.append({
            "agreeing_samples": votes,
            "out_of": results[0]["n"],
            "n_questions": len(results),
            "share_of_questions": len(results) / len(resultsById),
            "voted_ex": rightTotal / len(results),
        })

    # auroc of low agreement as a warning sign which is the chance a wrong answer had fewer votes than a right one
    wrongVotes = []
    rightVotes = []
    for result in resultsById.values():
        if result["votedEx"] == 1:
            rightVotes.append(result["winnerVotes"])
        else:
            wrongVotes.append(result["winnerVotes"])

    pairWins = 0.0
    for wrongValue in wrongVotes:
        for rightValue in rightVotes:
            if wrongValue < rightValue:
                pairWins = pairWins + 1
            elif wrongValue == rightValue:
                pairWins = pairWins + 0.5

    # undefined if every answer is right or every answer is wrong
    if len(wrongVotes) == 0 or len(rightVotes) == 0:
        agreementAuroc = float("nan")
    else:
        agreementAuroc = pairWins / (len(wrongVotes) * len(rightVotes))

    return pd.DataFrame(tableRows), agreementAuroc


def costTable(candidatesById: dict, seedSubset: list) -> dict:
    singleSeconds = 0.0
    totalSeconds = 0.0
    singleTokens = 0
    totalTokens = 0

    for candidates in candidatesById.values():
        for candidate in candidates:
            if candidate["seed"] not in seedSubset:
                continue
            totalSeconds = totalSeconds + candidate["row"]["amortized_sec"]
            totalTokens = totalTokens + candidate["row"]["n_output_tokens"]
            if candidate["seed"] == seedSubset[0]:
                singleSeconds = singleSeconds + candidate["row"]["amortized_sec"]
                singleTokens = singleTokens + candidate["row"]["n_output_tokens"]

    nQuestions = len(candidatesById)

    return {
        "n_samples": len(seedSubset),
        "single_sec_per_question": singleSeconds / nQuestions,
        "voted_sec_per_question": totalSeconds / nQuestions,
        "single_tokens_per_question": singleTokens / nQuestions,
        "voted_tokens_per_question": totalTokens / nQuestions,
        "cost_multiple": totalTokens / singleTokens,
    }


def exampleText(resultsById: dict, candidatesById: dict, wantFixed: bool, howMany: int) -> str:
    # a few concrete questions for the report showing what the vote changed
    lines = []
    shown = 0

    for devIdx in sorted(resultsById):
        result = resultsById[devIdx]
        isFixed = result["baselineEx"] == 0 and result["votedEx"] == 1
        isBroken = result["baselineEx"] == 1 and result["votedEx"] == 0

        if wantFixed and not isFixed:
            continue
        if not wantFixed and not isBroken:
            continue

        candidates = candidatesById[devIdx]
        baselineRow = candidates[0]["row"]

        chosenRow = None
        for candidate in candidates:
            if candidate["seed"] == result["chosenSeed"]:
                chosenRow = candidate["row"]

        # which seeds agreed with which so the reader can see the vote
        signatureLabels = {}
        voteParts = []
        for candidate in candidates:
            if candidate["signature"] is None:
                voteParts.append(f"{candidate['seed']}:failed")
                continue
            if candidate["signature"] not in signatureLabels:
                signatureLabels[candidate["signature"]] = chr(ord("A") + len(signatureLabels))
            label = signatureLabels[candidate["signature"]]
            voteParts.append(f"{candidate['seed']}:{label}({candidate['tokens']})")

        lines.append(f"**dev {devIdx} ({baselineRow['difficulty']})** {baselineRow['question']}")
        lines.append(f"- gold: `{baselineRow['gold']}`")
        lines.append(f"- baseline seed 42: `{baselineRow['pred_sql']}`")
        lines.append(f"- voted seed {result['chosenSeed']}: `{chosenRow['pred_sql']}`")
        lines.append(f"- votes by result with reasoning tokens: {' '.join(voteParts)}")
        lines.append("")

        shown = shown + 1
        if shown >= howMany:
            break

    if shown == 0:
        return "_none_\n"

    return "\n".join(lines) + "\n"


def toMarkdown(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))

    # object dtype keeps whole numbers as whole numbers instead of pandas turning a row of numbers into floats
    for _, tableRow in table.astype(object).iterrows():
        cells = []
        for column in columns:
            value = tableRow[column]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default = defaultBase, help = "the seed 42 run whose seeds get voted over")
    parser.add_argument("--seeds", nargs = "+", type = int, default = defaultSeeds, help = "seeds in the fixed tie breaking order")
    parser.add_argument("--writeN", nargs = "*", type = int, default = [3, 5], help = "sample counts to write run files for")
    args = parser.parse_args()

    # the baseline has to lead since ties and total failures fall back to it
    if args.seeds[0] != 42:
        print("seed 42 has to come first since its the baseline")
        return

    runsBySeed, usedSeeds = loadCandidateRuns(args.base, args.seeds)
    if 42 not in usedSeeds or len(usedSeeds) < 2:
        print("need the seed 42 run and at least one more scored seed")
        return

    print(f"voting over seeds {usedSeeds}")
    consistencyDir.mkdir(parents = True, exist_ok = True)

    candidatesById = buildCandidateTable(runsBySeed, usedSeeds)

    # every seed available is the main result
    resultsById = voteOverSeeds(candidatesById, usedSeeds)

    detailRows = []
    for devIdx, result in resultsById.items():
        detailRow = {"dev_idx": devIdx, "difficulty": candidatesById[devIdx][0]["row"]["difficulty"]}
        detailRow.update(result)
        detailRows.append(detailRow)
    pd.DataFrame(detailRows).to_csv(consistencyDir / f"votes_n{len(usedSeeds)}.csv", index = False)

    # paraphrase groups for the gain intervals
    groupKeyById = {}
    for devIdx, candidates in candidatesById.items():
        groupKeyById[devIdx] = goldGroupKey(candidates[0]["row"])

    mainResult = mainTable(resultsById, groupKeyById)

    # how many questions a tie actually decided which bounds how much the tie rule can matter
    tieCount = 0
    for result in resultsById.values():
        if result["decidedByTie"]:
            tieCount = tieCount + 1

    # the same vote with the seed order tie break as a sensitivity check
    seedOrderResults = voteOverSeeds(candidatesById, usedSeeds, tieBreak = "seedOrder")
    seedOrderResult = mainTable(seedOrderResults, groupKeyById)

    # the same vote but seed 42s query is kept whenever its in the winning group which removes the scorer quirk
    keepBaselineResult = mainTable(resultsById, groupKeyById, exKey = "votedExKeepBaseline")
    byDifficulty = difficultyTable(resultsById, candidatesById)
    scaling = scalingTable(candidatesById, usedSeeds)
    agreement, agreementAuroc = agreementTable(resultsById)
    cost = costTable(candidatesById, usedSeeds)

    byDifficulty.to_csv(consistencyDir / "by_difficulty.csv", index = False)
    scaling.to_csv(consistencyDir / "scaling.csv", index = False)
    agreement.to_csv(consistencyDir / "agreement.csv", index = False)

    # run files for scoring so the voted answers go through the official cross check like everything else
    writtenNames = []
    for subsetSize in args.writeN:
        if subsetSize > len(usedSeeds):
            print(f"not writing N={subsetSize} yet since only {len(usedSeeds)} seeds are scored")
            continue
        writtenNames.append(writeVotedRun(args.base, candidatesById, usedSeeds[:subsetSize], runsBySeed))

    seedTexts = []
    for seed in usedSeeds:
        seedTexts.append(str(seed))

    sections = []
    sections.append(f"# Self consistency for {args.base} over seeds {' '.join(seedTexts)}\n")
    sections.append("\n## Main result\n")
    sections.append(toMarkdown(pd.DataFrame([mainResult])))
    sections.append(f"\nquestions decided by a tie {tieCount} of {len(resultsById)}\n")
    sections.append("\n## Sensitivity with ties going to the earliest seed instead\n")
    sections.append(toMarkdown(pd.DataFrame([seedOrderResult])))
    sections.append("\n## Sensitivity keeping the baseline query whenever its in the winning group\n")
    sections.append("ex only changes here when the vote switches to a different answer so the scorers DISTINCT quirk cant flip it\n\n")
    sections.append(toMarkdown(pd.DataFrame([keepBaselineResult])))
    sections.append("\n## By difficulty\n")
    sections.append(toMarkdown(byDifficulty))
    sections.append("\n## Scaling averaged over every subset of seeds\n")
    sections.append(toMarkdown(scaling))
    sections.append("\n## Agreement as a confidence signal\n")
    sections.append(toMarkdown(agreement))
    sections.append(f"\nauroc of low agreement for flagging wrong answers {agreementAuroc:.3f}\n")
    sections.append("\n## Cost\n")
    sections.append(toMarkdown(pd.DataFrame([cost])))
    sections.append("\n## Questions the vote fixed\n")
    sections.append(exampleText(resultsById, candidatesById, wantFixed = True, howMany = 3))
    sections.append("\n## Questions the vote broke\n")
    sections.append(exampleText(resultsById, candidatesById, wantFixed = False, howMany = 3))

    (consistencyDir / "selfConsistency.md").write_text("\n".join(sections))

    print(
        f"voted ex {mainResult['voted_ex']:.3f} vs baseline {mainResult['baseline_ex']:.3f} "
        f"so a gain of {mainResult['gain']:+.3f} with 95 percent interval "
        f"{mainResult['gain_ci_low']:+.3f} to {mainResult['gain_ci_high']:+.3f} "
        f"and upper bound {mainResult['upper_bound_any_right']:.3f}"
    )
    print(
        f"keeping the baseline query inside its group gives {keepBaselineResult['voted_ex']:.3f} "
        f"and {mainResult['fixed_by_group_choice_only']} fixes plus {mainResult['broken_by_group_choice_only']} breaks came only from the group choice"
    )
    for votedName in writtenNames:
        print(f"wrote outputs/runs/{votedName}.jsonl so score it next")
    print(f"analysis in {consistencyDir / 'selfConsistency.md'}")


if __name__ == "__main__":
    main()