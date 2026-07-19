"""Solving hard problems by writing and running Python.

You are a careful problem solver and an expert Python programmer. You answer by
writing code, not by reasoning in prose alone: problems that are error-prone to
work out by hand are often easy to brute-force or verify with a short program.
"""

import argparse
import collections.abc

from effectful.handlers.llm import Template


@Template.define
def musr_object_placement(
    story: str, person: str, item: str, locations: collections.abc.Sequence[str]
) -> str:
    """A MuSR object-placement question: a theory-of-mind puzzle. Read the story
    and decide, from {locations}, where {person} would look for the {item}.

    The answer is the last place {person} *saw* the {item}: the last move they
    watched, or any later moment they directly saw it somewhere; or its original
    location if they never saw it after that. A person's belief does not change
    while they are not watching, so where the {item} actually ends up and where
    {person} believes it is can differ.

    <story>{story}</story>

    >>> musr_object_placement(
    ...     "Danny set the earphones in the recording booth, then stepped out for a "
    ...     "call. While he was gone, Emma quietly moved them to the producer's desk.",
    ...     "Danny",
    ...     "earphones",
    ...     ["recording booth", "producer's desk"],
    ... )
    'recording booth'
    """


def main() -> None:
    STUDIO_STORY = """\
In the heart of the bustling studio, Ricky, Emma, and Danny readied themselves \
for a day of creating magic. Ricky, the gifted singer-songwriter, had his \
precious notebook of lyrics on the producer's desk. Emma, their producer, was \
cognizant of the notebook's place at her desk. Across the room, Danny, the studio \
assistant, kept the earphones in the recording booth. They were all aware of the \
arrangement -- the notebook on the producer's desk, the earphones in the \
recording booth.

Ricky gently places his notebook onto the piano, then becomes engrossed in \
perfecting his song. Emma, engrossed in her thoughts, deftly moves the earphones \
to the producer's desk. At that moment Danny was in a stirring conversation with a \
visiting sound engineer; the visitor stood blocking Danny's general overview of \
the studio space.

Later, delicately lifting Ricky's notebook, Danny orchestrates its move to the \
producer's desk. At the desk, he glimpses a pair of earphones indirectly drawing \
his attention amidst his routine of tidying up. Meanwhile Emma, from inside a \
sound-proofed booth, was lost in reviewing already-recorded tracks, out of \
Danny's view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--story",
        type=str,
        default=STUDIO_STORY,
        help="The narrative describing where the item is moved and who saw it",
    )
    parser.add_argument(
        "--person",
        type=str,
        default="Danny",
        help="The person whose belief about the item's location is queried",
    )
    parser.add_argument(
        "--item",
        type=str,
        default="earphones",
        help="The object being tracked",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["piano", "producer's desk", "recording booth"],
        metavar="LOCATION",
        help="Candidate locations to choose the answer from",
    )
    args = parser.parse_args()

    answer = musr_object_placement(args.story, args.person, args.item, args.locations)
    print(f"MuSR: where would {args.person} look for the {args.item}?")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
