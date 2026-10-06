"""Curated list of public-domain Project Gutenberg books used to build a
demo/sample Calibre library.

This is deliberately a mix of overlapping authors (Dickens, Wells, Conan
Doyle) and thematically adjacent books (Frankenstein vs. Jekyll & Hyde,
two Sherlock Holmes novels) so that the "fuzzy memory" search has
genuinely ambiguous cases to disambiguate — a good demo of the LLM
re-ranking step, not just vector similarity.

``comments`` are short original blurbs written for this project (not
copied from Gutenberg), phrased the way a reader might half-remember a
book, which is exactly the kind of text we want in the index.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SampleBook:
    gutenberg_id: int
    title: str
    authors: list[str]
    tags: list[str]
    comments: str
    series: tuple[str, int] | None = field(default=None)
    pubdate: str = "1900-01-01"

    @property
    def gutenberg_txt_url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gutenberg_id}/pg{self.gutenberg_id}.txt"


SAMPLE_BOOKS: list[SampleBook] = [
    SampleBook(
        gutenberg_id=1342,
        title="Pride and Prejudice",
        authors=["Jane Austen"],
        tags=["Classic Literature", "Romance", "Satire"],
        comments=(
            "A sharp-tongued young woman clashes with a proud, wealthy gentleman at "
            "a country dance, and spends the rest of the book realising she "
            "misjudged him completely."
        ),
        pubdate="1813-01-28",
    ),
    SampleBook(
        gutenberg_id=84,
        title="Frankenstein; Or, The Modern Prometheus",
        authors=["Mary Shelley"],
        tags=["Gothic", "Horror", "Science Fiction"],
        comments=(
            "A student obsessed with the secret of life assembles a creature from "
            "dead flesh, then abandons it in horror — and the creature comes looking "
            "for him."
        ),
        pubdate="1818-01-01",
    ),
    SampleBook(
        gutenberg_id=345,
        title="Dracula",
        authors=["Bram Stoker"],
        tags=["Gothic", "Horror"],
        comments=(
            "Told entirely through letters and diary entries, a young solicitor "
            "travels to a remote Transylvanian castle and slowly realises his "
            "host isn't human."
        ),
        pubdate="1897-05-26",
    ),
    SampleBook(
        gutenberg_id=2701,
        title="Moby-Dick; Or, The Whale",
        authors=["Herman Melville"],
        tags=["Classic Literature", "Adventure", "Sea Stories"],
        comments=(
            "A one-legged sea captain drags his crew across the oceans in a "
            "monomaniacal hunt for the enormous white whale that took his leg."
        ),
        pubdate="1851-10-18",
    ),
    SampleBook(
        gutenberg_id=11,
        title="Alice's Adventures in Wonderland",
        authors=["Lewis Carroll"],
        tags=["Children's Literature", "Fantasy", "Satire"],
        comments=(
            "A girl follows a waistcoat-wearing rabbit down a hole and ends up at a "
            "tea party with a vanishing cat, a hookah-smoking caterpillar, and a "
            "furious card-playing queen."
        ),
        pubdate="1865-11-26",
    ),
    SampleBook(
        gutenberg_id=1661,
        title="The Adventures of Sherlock Holmes",
        authors=["Arthur Conan Doyle"],
        tags=["Mystery", "Detective Fiction", "Classic Literature"],
        comments=(
            "A collection of short cases for the detective at 221B Baker Street, "
            "narrated by his loyal, perpetually one-step-behind friend and flatmate."
        ),
        series=("Sherlock Holmes", 3),
        pubdate="1892-10-14",
    ),
    SampleBook(
        gutenberg_id=244,
        title="A Study in Scarlet",
        authors=["Arthur Conan Doyle"],
        tags=["Mystery", "Detective Fiction", "Classic Literature"],
        comments=(
            "The first case: a doctor just back from war in Afghanistan moves in "
            "with an eccentric new flatmate who can deduce your entire life story "
            "from your shoes."
        ),
        series=("Sherlock Holmes", 1),
        pubdate="1887-11-01",
    ),
    SampleBook(
        gutenberg_id=1400,
        title="Great Expectations",
        authors=["Charles Dickens"],
        tags=["Classic Literature", "Coming of Age"],
        comments=(
            "An orphan boy who helps an escaped convict on the marshes is later "
            "given a mysterious fortune by an anonymous benefactor, and grows up "
            "ashamed of where he came from."
        ),
        pubdate="1861-08-01",
    ),
    SampleBook(
        gutenberg_id=98,
        title="A Tale of Two Cities",
        authors=["Charles Dickens"],
        tags=["Classic Literature", "Historical Fiction"],
        comments=(
            "Set during the French Revolution, a man who looks exactly like a "
            "condemned prisoner makes a final, selfless choice at the guillotine."
        ),
        pubdate="1859-04-30",
    ),
    SampleBook(
        gutenberg_id=174,
        title="The Picture of Dorian Gray",
        authors=["Oscar Wilde"],
        tags=["Gothic", "Classic Literature", "Philosophical Fiction"],
        comments=(
            "A beautiful young man wishes his portrait would age instead of him — "
            "and gets his wish, while the painting in the attic grows more hideous "
            "with every sin."
        ),
        pubdate="1890-07-01",
    ),
    SampleBook(
        gutenberg_id=43,
        title="The Strange Case of Dr. Jekyll and Mr. Hyde",
        authors=["Robert Louis Stevenson"],
        tags=["Gothic", "Horror", "Mystery"],
        comments=(
            "A respected London doctor develops a potion that splits him into a "
            "second, monstrous identity he can no longer fully control."
        ),
        pubdate="1886-01-05",
    ),
    SampleBook(
        gutenberg_id=35,
        title="The Time Machine",
        authors=["H. G. Wells"],
        tags=["Science Fiction", "Classic Literature"],
        comments=(
            "An inventor travels hundreds of thousands of years into the future "
            "and finds humanity split into two very different descendant species."
        ),
        pubdate="1895-05-07",
    ),
    SampleBook(
        gutenberg_id=36,
        title="The War of the Worlds",
        authors=["H. G. Wells"],
        tags=["Science Fiction", "Classic Literature"],
        comments=(
            "Martian tripods land in the English countryside and begin methodically "
            "destroying everything in their path with a heat-ray."
        ),
        pubdate="1898-01-01",
    ),
    SampleBook(
        gutenberg_id=219,
        title="Heart of Darkness",
        authors=["Joseph Conrad"],
        tags=["Classic Literature", "Adventure", "Philosophical Fiction"],
        comments=(
            "A steamboat captain travels up an African river to retrieve a company "
            "agent who has gone native and set himself up as a god among the locals."
        ),
        pubdate="1899-02-01",
    ),
    SampleBook(
        gutenberg_id=514,
        title="Little Women",
        authors=["Louisa May Alcott"],
        tags=["Classic Literature", "Coming of Age", "Family"],
        comments=(
            "Four sisters growing up during the Civil War navigate first loves, "
            "ambitions, and loss, while their father is away at the front."
        ),
        pubdate="1868-09-30",
    ),
    SampleBook(
        gutenberg_id=55,
        title="The Wonderful Wizard of Oz",
        authors=["L. Frank Baum"],
        tags=["Children's Literature", "Fantasy"],
        comments=(
            "A tornado sweeps a farm girl and her dog into a magical land, where "
            "she follows a road of yellow bricks to find a wizard who can send her "
            "home."
        ),
        pubdate="1900-05-01",
    ),
    SampleBook(
        gutenberg_id=74,
        title="The Adventures of Tom Sawyer",
        authors=["Mark Twain"],
        tags=["Classic Literature", "Coming of Age", "Adventure"],
        comments=(
            "A mischievous boy in a small Mississippi river town tricks his friends "
            "into whitewashing a fence for him, then witnesses a murder in a "
            "graveyard at midnight."
        ),
        pubdate="1876-06-01",
    ),
    SampleBook(
        gutenberg_id=164,
        title="Twenty Thousand Leagues Under the Sea",
        authors=["Jules Verne"],
        tags=["Science Fiction", "Adventure"],
        comments=(
            "A marine biologist is taken captive aboard a fantastical electric "
            "submarine by a brooding captain who has cut all ties to the surface "
            "world."
        ),
        series=("Voyages Extraordinaires", 6),
        pubdate="1870-06-20",
    ),
    SampleBook(
        gutenberg_id=45,
        title="Anne of Green Gables",
        authors=["L. M. Montgomery"],
        tags=["Classic Literature", "Coming of Age", "Family"],
        comments=(
            "An elderly brother and sister accidentally adopt a talkative orphan "
            "girl instead of the boy they asked for, and she talks her way into "
            "their hearts anyway."
        ),
        series=("Anne of Green Gables", 1),
        pubdate="1908-06-01",
    ),
    SampleBook(
        gutenberg_id=5200,
        title="The Metamorphosis",
        authors=["Franz Kafka"],
        tags=["Philosophical Fiction", "Horror"],
        comments=(
            "A travelling salesman wakes up one morning transformed into a giant "
            "insect, and his family slowly stops seeing him as family."
        ),
        pubdate="1915-01-01",
    ),
]
