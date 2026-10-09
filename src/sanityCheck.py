"""
checks to run before trusting any model results
  1  sql extraction handles the reply formats we expect
  2  scoring the gold queries against themselves gives 100 percent on em and ex and ex with gold values from our scorer and the official one
  3  replies with nothing extracted get scored wrong the same way by both

how to run
    python src/sanityCheck.py
"""

from scoring import crossCheckWithOfficial, printSummary, scoreExample, summarizeScores
from spiderUtils import extractSql, outDir, readJsonl


def checkExtraction():
    # each case is a reply and whether its a reasoning model and the sql and status we expect back
    # covers the formats seen in the pilots plus the failure cases
    testCases = [
        (
            "Here you go:\n```sql\nSELECT name\nFROM singer;\n```",
            False,
            "SELECT name FROM singer",
            "code_block",
        ),
        (
            "```sql\nSELECT 1\n```\nActually:\n```sql\nSELECT count(*) FROM singer\n```",
            False,
            "SELECT count(*) FROM singer",
            "code_block",
        ),
        (
            "SELECT name FROM singer WHERE age > 30;",
            False,
            "SELECT name FROM singer WHERE age > 30",
            "fallback",
        ),
        (
            "<think>Maybe SELECT * FROM singer? Let me check the schema</think>\n"
            "```sql\nSELECT name FROM singer\n```",
            True,
            "SELECT name FROM singer",
            "code_block",
        ),
        (
            "<think>I need the singer table, SELECT name FROM",
            True,
            None,
            "truncated_in_reasoning",
        ),
        (
            "I cannot answer this.",
            False,
            None,
            "no_sql",
        ),
    ]

    for rawText, isReasoning, expectedSql, expectedStatus in testCases:
        result = extractSql(rawText, isReasoning = isReasoning)
        expected = (expectedSql, expectedStatus)

        assert result == expected, f"\n{rawText!r}\n got {result}\n want {expected}"

    print(f"extraction all {len(testCases)} cases passed")


def scoreRows(rows: list) -> list:
    for row in rows:
        scores = scoreExample(row["pred_sql"], row["gold"], row["db_id"])
        row.update(scores)

    return rows


def checkGoldAgainstGold(subsetRows: list):
    # the gold query graded against itself has to be right every time
    # anything less means the scorer or the data is broken
    rows = []
    for subsetRow in subsetRows:
        row = dict(subsetRow)
        row["pred_sql"] = subsetRow["gold"]
        rows.append(row)

    rows = scoreRows(rows)
    printSummary(summarizeScores(rows), title = "gold vs gold expecting 1.000 everywhere")

    wrongIds = []
    for row in rows:
        if row["em"] != 1 or row["ex"] != 1 or row["ex_gold_values"] != 1:
            wrongIds.append(row["dev_idx"])

    assert len(wrongIds) == 0, f"gold scored wrong on dev_idx {wrongIds}"

    crossCheckWithOfficial(rows, "sanity_gold")


def checkInvalidOutputs(subsetRows: list):
    # blank out a few predictions and the score has to drop by exactly that many
    # proves missing answers count as wrong and that the official script agrees
    nBlanked = 5

    rows = []
    for position, subsetRow in enumerate(subsetRows):
        row = dict(subsetRow)
        if position < nBlanked:
            row["pred_sql"] = None
        else:
            row["pred_sql"] = subsetRow["gold"]
        rows.append(row)

    rows = scoreRows(rows)
    overall = summarizeScores(rows)["all"]

    expectedScore = 1 - nBlanked / len(rows)

    # tiny tolerance just for floating point
    assert abs(overall["em"] - expectedScore) < 1e-9
    assert abs(overall["ex"] - expectedScore) < 1e-9
    assert abs(overall["ex_gold_values"] - expectedScore) < 1e-9

    crossCheckWithOfficial(rows, "sanity_invalid")

    print(f"invalid output handling with {nBlanked} blanked gave EM and EX and EX with gold values of {expectedScore:.3f} as expected")


def main():
    checkExtraction()

    subsetRows = readJsonl(outDir / "subset.jsonl")

    checkGoldAgainstGold(subsetRows)
    checkInvalidOutputs(subsetRows)

    print("\nall sanity checks passed")


if __name__ == "__main__":
    main()