"""A small, hand-written evaluation set.

Each case is a "fuzzy memory" query phrased the way a real reader would
describe a half-remembered book, paired with the title we expect the agent
to land on. This is deliberately tiny and readable rather than a large
automated benchmark — the point is to demonstrate the *practice* of
evaluating LLM output against ground truth, not to be exhaustive.

Queries intentionally omit the title/author to force the agent to rely on
plot/character/vibe matching (i.e. exercise the same path a real "I can't
remember what it's called" query would take), and some are deliberately
ambiguous between two of the sample library's books to test that the agent
disambiguates using later tool calls / re-ranking rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_title: str


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        query="A scientist builds a creature out of dead body parts and then is horrified by it",
        expected_title="Frankenstein; Or, The Modern Prometheus",
    ),
    EvalCase(
        query=(
            "There's a detective who lives on Baker Street with a doctor who writes down his cases"
        ),
        expected_title="The Adventures of Sherlock Holmes",
    ),
    EvalCase(
        query="A girl falls down a rabbit hole and meets a grinning cat and a mad tea party",
        expected_title="Alice's Adventures in Wonderland",
    ),
    EvalCase(
        query="Martians invade England with heat rays and walking machines",
        expected_title="The War of the Worlds",
    ),
    EvalCase(
        query="A respectable doctor drinks a potion and turns into someone evil at night",
        expected_title="The Strange Case of Dr. Jekyll and Mr. Hyde",
    ),
    EvalCase(
        query=(
            "An orphan boy helps an escaped convict and later gets money from a secret benefactor"
        ),
        expected_title="Great Expectations",
    ),
    EvalCase(
        query="A ship captain is obsessed with hunting one specific enormous white whale",
        expected_title="Moby-Dick; Or, The Whale",
    ),
    EvalCase(
        query="A man's portrait in his attic ages and rots instead of him while he stays young",
        expected_title="The Picture of Dorian Gray",
    ),
    EvalCase(
        query="A tornado takes a farm girl to a magical land with a yellow brick road",
        expected_title="The Wonderful Wizard of Oz",
    ),
    EvalCase(
        query="A man wakes up one day having turned into a giant insect",
        expected_title="The Metamorphosis",
    ),
]
