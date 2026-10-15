"""Self-authored and public-domain text stimuli for the language tests.

Licensing position (this is the important part)
----------------------------------------------
Every string in this bank is either

1. **written for this repository** (sentences, word lists, pseudowords, digits,
   arithmetic, questions, passages and the N400 material), or
2. **public domain by age**: Aesop's fables (traditional, >2000 years old;
   re-told here in our own words), Lewis Carroll's *You Are Old, Father William*
   and *Jabberwocky* (1855/1871), and Edward Lear's *The Owl and the Pussycat*
   (1871).

No copyrighted prose is copied.  Each passage carries ``meta['license']`` so
that a published dataset can state this per item, and the module docstring is
the authoritative statement of provenance.

Why a JSON file at all
----------------------
The specs reference categories by *name* (``textbank.items('passages_hard')``),
and the analysis report needs the exact text that was shown.  Writing the bank
once to ``data/stimuli/text/textbank.json`` gives the report a stable artefact,
lets an operator edit the last few items (or the ``own_name`` placeholder)
without touching code, and makes the whole set inspectable in a diff.
Generation is seeded, so the file is byte-stable across runs unless ``force``.

Design notes that matter for the analyses
-----------------------------------------
* ``passages_easy/medium/hard`` differ in *syntactic and lexical* difficulty
  (short common-word clauses -> subordinate clauses -> abstract nominalisations),
  not just in length, because T507 is a readability ladder.
* ``n400_congruent`` / ``n400_incongruent`` are the *same sentence frames* with a
  plausible vs implausible final word; only the final word changes, so the
  comparison is a clean semantic violation (T404).
* ``words_high_freq`` are among the most frequent English nouns; ``words_low_freq``
  are rare but real and imageable (``quiver``, ``anvil``), so T503 measures
  frequency and not word-likeness.
* ``pseudowords`` are phonotactically legal English nonwords built from
  onsets/nuclei/codas; a lexical decision (T406) is then a real word/nonword
  judgement rather than a spelling test.
* ``shuffled_word_lists`` contain *exactly the words* of ``prose_passages`` in a
  different order, so T505 contrasts structure and not content.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import TEXT_DIR, ensure_dirs

#: Where the generated bank lives.
TEXTBANK_FILE = TEXT_DIR / "textbank.json"

#: Categories every test may ask for.  ``passages`` is an alias category (see
#: :data:`ALIASES`) requested by T501; ``pseudowords`` is also produced by
#: :func:`pseudowords`.
REQUIRED_CATEGORIES = [
    "simple_sentences", "garden_path_sentences", "n400_congruent",
    "n400_incongruent", "words_high_freq", "words_low_freq", "pseudowords",
    "passages_easy", "passages_medium", "passages_hard", "passages_positive",
    "passages_negative", "passages_neutral", "prose_passages", "shuffled_word_lists",
    "digits", "stroop_congruent", "stroop_incongruent", "nback_letters",
    "arithmetic_easy", "arithmetic_hard", "own_name", "other_names",
    "names_common", "questions_yes_no", "read_aloud_sentences",
]

#: Names that resolve to another category.  T501 asks for ``passages`` with no
#: qualifier; rather than duplicating prose we point it at ``passages_medium``,
#: which is the neutral middle of the readability ladder.
ALIASES: dict[str, str] = {"passages": "passages_medium", "passage": "passages_medium",
                           "prose": "prose_passages", "shuffled": "shuffled_word_lists"}

DEFAULT_SEED = 20261013

#: Colour words and ink names for the Stroop tests.  ``stroop_congruent`` items
#: have word == ink; ``stroop_incongruent`` items always name a *different*
#: colour, which is the source of the interference in T203.
COLOURS = ("red", "blue", "green", "yellow", "purple", "orange")

#: Visual ink colours as hex strings, so the runner does not have to guess what
#: "red" means on a dark background.
COLOUR_HEX = {"red": "#e02020", "blue": "#2050e0", "green": "#20a040",
              "yellow": "#e0c020", "purple": "#9030c0", "orange": "#e08020"}


@dataclass
class TextItem:
    """One presentable text stimulus (word, sentence or passage)."""

    text: str
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- content
# Public-domain verse, cited exactly so the licence claim is checkable.
_CARROLL_FATHER_WILLIAM = (
    '"You are old, Father William," the young man said,\n'
    '"And your hair has become very white;\n'
    'And yet you incessantly stand on your head --\n'
    'Do you think, at your age, it is right?"\n\n'
    '"In my youth," Father William replied to his son,\n'
    '"I feared it might injure the brain;\n'
    'But, now that I\'m perfectly sure I have none,\n'
    'Why, I do it again and again."'
)
_CARROLL_JABBERWOCKY = (
    "'Twas brillig, and the slithy toves\n"
    'Did gyre and gimble in the wabe:\n'
    'All mimsy were the borogoves,\n'
    'And the mome raths outgrabe.'
)
_LEAR_OWL = (
    'The Owl and the Pussy-cat went to sea\n'
    'In a beautiful pea-green boat,\n'
    'They took some honey, and plenty of money,\n'
    'Wrapped up in a five-pound note.'
)

# Self-authored prose.  Kept short (60-110 words) so a "passage" fits the
# self-paced reading design without scrolling on a 1280x800 display.
_PROSE: list[tuple[str, str]] = [
    ("aesop_north_wind_sun",
     "The North Wind and the Sun argued about which of them was the stronger. "
     "They agreed to settle the matter on a traveller walking along the road below. "
     "The Wind blew first, hard and cold, but the harder it blew the more tightly the "
     "man wrapped his coat around himself. Then the Sun shone gently, and the traveller "
     "loosened his collar, sat down in the shade, and took the coat off. Persuasion had "
     "done what force could not."),
    ("aesop_fox_grapes",
     "A hungry fox found a vine heavy with ripe grapes. The fruit hung high above the "
     "ground, and though he leapt again and again he could not reach it. At last he turned "
     "away and said that the grapes were surely sour and not worth eating. It is easy to "
     "despise what we cannot have, and easier still to say so out loud."),
    ("aesop_tortoise_hare",
     "A hare mocked a tortoise for the slowness of his walk. The tortoise answered that "
     "he would race the hare to the oak at the top of the hill. The hare ran far ahead, "
     "then stopped to rest in the warm grass and fell asleep. The tortoise kept walking, "
     "slow and steady, and reached the oak first. Slow but constant work finishes what "
     "speed alone does not."),
    ("aesop_lion_mouse",
     "A mouse ran across the paws of a sleeping lion. The lion woke, caught him, and was "
     "about to eat him when the mouse begged to be spared. The lion laughed and let him go. "
     "Not long after, the lion was caught in a hunter's net, and the mouse gnawed through "
     "the ropes until the great beast was free. Small help given freely is still help."),
    ("aesop_ant_grasshopper",
     "All summer the ants carried grain into their nest while a grasshopper sang in the "
     "sun. When the cold came, the grasshopper had nothing stored and begged the ants for "
     "food. They asked him what he had done all summer. He said that he had sung. Then "
     "they told him that since he had sung all summer, he might dance all winter."),
    ("selfauthored_kitchen",
     "The kitchen was warm and smelled of bread. A kettle hummed on the back ring of the "
     "stove, and steam gathered on the window above the sink. On the table lay a knife, a "
     "loaf, and a bowl of apples. Nothing in the room moved except the steam and a fly near "
     "the light. It was the quiet hour before the household woke, when the day has not yet "
     "begun to make its demands."),
    ("selfauthored_station",
     "Trains leave this station every twenty minutes, and the board above the platform "
     "lists them in orange letters that flicker when the wind moves the cables. People wait "
     "in loose groups, each group around its own suitcase. A guard walks the length of the "
     "platform and looks at nothing in particular. When a train arrives, the groups fold "
     "into the doors, the doors close, and the platform is empty again for a little while."),
    ("carroll_father_william", _CARROLL_FATHER_WILLIAM),
    ("carroll_jabberwocky", _CARROLL_JABBERWOCKY),
    ("lear_owl_and_pussycat", _LEAR_OWL),
]

#: Word-frequency contrast for T503.  ``words_high_freq`` are drawn from the most
#: frequent English nouns; ``words_low_freq`` are rare but valid and imageable.
_WORDS_HIGH = [
    "time", "people", "water", "house", "money", "night", "place", "world",
    "hand", "book", "friend", "family", "school", "story", "music", "letter",
    "garden", "window", "table", "street", "morning", "summer", "mother",
    "father", "child", "train", "market", "picture", "answer", "question",
]
_WORDS_LOW = [
    "anvil", "quiver", "lattice", "marmot", "vellum", "gannet", "trellis",
    "kestrel", "obsidian", "fathom", "penumbra", "saffron", "juniper",
    "abacus", "ferrule", "gudgeon", "harrier", "kelpie", "loam", "mews",
    "nadir", "oxbow", "plinth", "quagga", "rowan", "selvage", "tarn",
    "umbel", "vane", "wicket",
]

#: Pool used to finish the N400 frames: concrete, high-frequency nouns that are
#: strongly associated with exactly one frame.
_N400_FRAMES: list[tuple[str, str, str]] = [
    ("She spread butter on her slice of", "bread", "ceiling"),
    ("He poured milk into his cup of", "coffee", "stairs"),
    ("The gardener watered the", "flowers", "sentence"),
    ("She tied the laces on her", "shoes", "clouds"),
    ("He rowed the small boat across the", "lake", "carpet"),
    ("The chef sharpened his kitchen", "knife", "window"),
    ("She brushed her hair with a", "comb", "ladder"),
    ("The child built a tower of", "blocks", "thunder"),
    ("He read the news in the morning", "paper", "garden"),
    ("They planted seeds in the", "soil", "engine"),
    ("She locked the door with a", "key", "song"),
    ("The cat slept on the warm", "sofa", "river"),
    ("He hung his coat on the", "hook", "idea"),
    ("The bird built a nest in the", "tree", "kettle"),
    ("She wrote the letter with a", "pen", "stone"),
    ("They crossed the river by the", "bridge", "silence"),
    ("The farmer milked the", "cow", "piano"),
    ("He sliced the loaf with a", "knife", "mirror"),
    ("She lit the candle with a", "match", "harbour"),
    ("The waiter carried a tray of", "glasses", "mountains"),
]

#: Garden-path sentences with a *temporary* ambiguity: the first pass parses
#: them wrongly (usually as a reduced relative clause), and the disambiguating
#: word forces a re-analysis.  T504 contrasts these with the simple sentences.
_GARDEN_PATH = [
    "The old man the boat.",
    "The horse raced past the barn fell.",
    "The cotton clothing is made of grows in Mississippi.",
    "While Anna dressed the baby played in the crib.",
    "The complex houses married and single students.",
    "The doctor told the patient he had been seeing for years would recover.",
    "The prime number few people understand is seven.",
    "The government plans to raise taxes were defeated.",
    "The fisherman left the dock wet and cold.",
    "When the car crashed the driver had been drinking heavily.",
    "The raft floated down the river sank near the bend.",
    "The editor sent the writer the letter praised by everyone.",
]

_SIMPLE_SENTENCES = [
    "The dog sleeps on the mat.",
    "She drinks a glass of water.",
    "The sun rises in the east.",
    "He opens the red door.",
    "The children play in the park.",
    "A bird sings in the tree.",
    "The train leaves at nine.",
    "She reads a short book.",
    "The cat sits by the fire.",
    "He carries a heavy bag.",
    "The rain falls on the roof.",
    "They walk to the corner shop.",
    "The kettle boils on the stove.",
    "A lamp lights the small room.",
    "She writes her name on the page.",
    "The bus stops near the school.",
    "He cuts the bread with a knife.",
    "The river runs past the farm.",
    "A clock ticks on the wall.",
    "They eat dinner at six.",
]

_READ_ALOUD = [
    "The quick brown fox jumps over the lazy dog.",
    "Please read this sentence in a clear and steady voice.",
    "The library closes at eight o'clock this evening.",
    "A steady rhythm helps the recording stay aligned.",
    "The mountain path was narrow but well marked.",
    "She counted the coins twice before putting them away.",
    "Bright light through the window woke the sleeping baby.",
    "The recipe calls for two eggs and a cup of flour.",
    "He repaired the broken fence with a length of wire.",
    "The audience waited quietly for the concert to begin.",
]

_QUESTIONS_YES_NO = [
    "Is the door closed?",
    "Did you eat breakfast today?",
    "Is it raining outside?",
    "Have you been to the coast?",
    "Is the light switched on?",
    "Did the train arrive on time?",
    "Are there any letters on the table?",
    "Is the window open?",
    "Did you sleep well last night?",
    "Is the kettle boiling?",
]

_OWN_NAME_PLACEHOLDER = "<<PARTICIPANT_NAME>>"

_OTHER_NAMES = [
    "Michael", "Sarah", "David", "Laura", "James", "Emma", "Thomas", "Julia",
    "Peter", "Rachel", "Daniel", "Hannah", "Simon", "Claire", "Andrew", "Maria",
]

_COMMON_NAMES = [
    "John", "Mary", "Robert", "Susan", "William", "Linda", "Richard", "Karen",
    "Charles", "Nancy", "Joseph", "Betty", "George", "Helen", "Edward", "Alice",
    "Paul", "Diane", "Mark", "Ruth",
]

_NBACK_LETTERS = list("BCDFGHJKLMNPQRSTVWXZ")


def _rng(seed: int | str | None = None) -> random.Random:
    """A local RNG: never touches the global random state (reproducibility).

    Accepts strings by hashing them, so a per-passage shuffle can be seeded with
    ``f"{seed}:{key}"`` and stay stable across runs and Python versions.
    """
    raw: Any = DEFAULT_SEED if seed is None else seed
    if isinstance(raw, str):
        raw = int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(int(raw))


# ---------------------------------------------------------------- generators
def _pseudoword(rng: random.Random) -> str:
    """Build one pronounceable, orthographically legal nonword."""
    from .synth import CODAS, NUCLEI, ONSETS

    n_syl = rng.choice([1, 2, 2, 2, 3])
    out: list[str] = []
    for i in range(n_syl):
        out.append(rng.choice(ONSETS))
        out.append(rng.choice(NUCLEI))
        if i == n_syl - 1:
            out.append(rng.choice(CODAS))
        elif rng.random() < 0.35:
            out.append(rng.choice([c for c in CODAS if c in ("l", "m", "n", "r", "s", "ck")]))
    word = "".join(out)
    # keep it word-shaped: 3-10 letters, starts with a consonant
    if len(word) < 3 or len(word) > 10:
        return _pseudoword(rng)
    return word


def pseudowords(n: int, seed: int = DEFAULT_SEED) -> list[str]:
    """``n`` distinct pronounceable nonwords (deterministic for ``seed``)."""
    rng = _rng(seed)
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < max(0, int(n)) and guard < 50 * max(1, int(n)) + 100:
        guard += 1
        w = _pseudoword(rng)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _nback_sequences(rng: random.Random, n_seq: int = 12, length: int = 16) -> list[str]:
    """Letter streams for the N-back task.

    Built as *one string of letters per sequence* (the runner re-uses a stream
    across trials): every 4th item is a repeat of the item 2 back, so a 2-back
    target occurs at a known rate and the remaining items are non-repeats.  The
    structure is what makes T201's ``load_high`` condition well-formed.
    """
    out: list[str] = []
    for _ in range(n_seq):
        letters: list[str] = [rng.choice(_NBACK_LETTERS)]
        while len(letters) < length:
            if len(letters) >= 2 and rng.random() < 0.3:
                letters.append(letters[-2])              # a 2-back target
            else:
                c = rng.choice(_NBACK_LETTERS)
                if len(letters) >= 2 and c == letters[-2]:
                    continue
                letters.append(c)
        out.append("".join(letters))
    return out


def _arithmetic_easy(rng: random.Random, n: int = 24) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for _ in range(n):
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        if rng.random() < 0.5:
            out.append((f"{a} + {b}", str(a + b)))
        else:
            hi, lo = max(a, b), min(a, b)
            out.append((f"{hi} - {lo}", str(hi - lo)))
    return out


def _arithmetic_hard(rng: random.Random, n: int = 24) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for _ in range(n):
        a, b = rng.randint(11, 39), rng.randint(3, 19)
        c = rng.randint(2, 9)
        kind = rng.choice(["mul_add", "mul_sub", "three_term"])
        if kind == "mul_add":
            out.append((f"{a} + {b} x {c}", str(a + b * c)))
        elif kind == "mul_sub":
            out.append((f"{a} - {b} x {c}", str(a - b * c)))
        else:
            d = rng.randint(2, 9)
            out.append((f"{a} + {b} - {d}", str(a + b - d)))
    return out


# ---------------------------------------------------------------- assembly
def _items_for(category: str, seed: int = DEFAULT_SEED) -> list[TextItem]:
    """Materialise one category (deterministic; the JSON is built from this)."""
    rng = _rng(seed)
    cat = category

    if cat == "simple_sentences":
        return [TextItem(t, "simple", {"words": len(t.split())}) for t in _SIMPLE_SENTENCES]

    if cat == "read_aloud_sentences":
        return [TextItem(t, "read_aloud", {"words": len(t.split())}) for t in _READ_ALOUD]

    if cat == "questions_yes_no":
        return [TextItem(t, "question", {"words": len(t.split())}) for t in _QUESTIONS_YES_NO]

    if cat == "garden_path_sentences":
        return [TextItem(t, "garden_path", {"words": len(t.split()),
                                            "ambiguity": "temporary"}) for t in _GARDEN_PATH]

    if cat in ("n400_congruent", "n400_incongruent"):
        # == not endswith(): "n400_incongruent".endswith("congruent") is True and
        # would fill the incongruent condition with congruent sentences.
        which = 2 if cat == "n400_incongruent" else 1
        out = []
        for stem, plausible, implausible in _N400_FRAMES:
            final = plausible if which == 1 else implausible
            out.append(TextItem(f"{stem} {final}.",
                                "congruent" if which == 1 else "incongruent",
                                {"frame": stem, "final_word": final,
                                 "cloze": "high" if which == 1 else "low"}))
        return out

    if cat == "words_high_freq":
        return [TextItem(w, "high_freq", {"frequency_class": "high"}) for w in _WORDS_HIGH]

    if cat == "words_low_freq":
        return [TextItem(w, "low_freq", {"frequency_class": "low"}) for w in _WORDS_LOW]

    if cat == "pseudowords":
        return [TextItem(w, "pseudoword", {"lexicality": "nonword"}) for w in pseudowords(40, seed)]

    if cat in ("passages_easy", "passages_medium", "passages_hard"):
        level = cat.split("_")[1]
        texts = _passages_for_level(level)
        return [TextItem(t, level, {"readability": level, "words": len(t.split()),
                                    "license": "self-authored / public domain"})
                for t in texts]

    if cat in ("passages_positive", "passages_negative", "passages_neutral"):
        tone = cat.split("_")[1]
        return [TextItem(t, tone, {"valence": tone, "words": len(t.split()),
                                   "license": "self-authored"})
                for t in _passages_for_tone(tone)]

    if cat == "prose_passages":
        return [TextItem(t, "prose", {"key": key, "words": len(t.split()),
                                      "license": _prose_license(key)})
                for key, t in _PROSE]

    if cat == "shuffled_word_lists":
        out = []
        for key, text in _PROSE:
            words = text.split()
            shuffled = list(words)
            # Seeded by the passage so the scramble is stable but differs from
            # the coherent order (the whole point of T505).
            local = _rng(f"{seed}:{key}")
            local.shuffle(shuffled)
            out.append(TextItem(" ".join(shuffled), "shuffled",
                                {"key": key, "source_key": key, "words": len(shuffled),
                                 "same_words_as": key}))
        return out

    if cat == "digits":
        out = [TextItem(str(d), "digit", {"value": d}) for d in range(10)]
        out += [TextItem(" ".join(str(rng.randint(0, 9)) for _ in range(12)), "digit_stream",
                         {"n": 12}) for _ in range(4)]
        return out

    if cat in ("stroop_congruent", "stroop_incongruent"):
        # NOTE: use ==, not endswith(): "stroop_incongruent".endswith("congruent")
        # is True, which silently produced congruent items in the incongruent
        # condition -- i.e. no interference, and a dead test.
        congruent = (cat == "stroop_congruent")
        out = []
        for word in COLOURS:
            if congruent:
                ink = word
            else:
                others = [c for c in COLOURS if c != word]
                ink = others[rng.randrange(len(others))]
            out.append(TextItem(word, "congruent" if congruent else "incongruent",
                                {"ink": ink, "ink_hex": COLOUR_HEX[ink], "word": word,
                                 "response": ink, "incongruent": not congruent}))
        # Repeat so there are enough trials for a block.
        base = list(out)
        for _ in range(3):
            extra = list(base)
            rng.shuffle(extra)
            out += extra
        return out

    if cat == "nback_letters":
        return [TextItem(seq, "nback", {"n": 2, "length": len(seq)})
                for seq in _nback_sequences(rng)]

    if cat == "arithmetic_easy":
        return [TextItem(q, "easy", {"answer": a, "difficulty": "easy"})
                for q, a in _arithmetic_easy(rng)]

    if cat == "arithmetic_hard":
        return [TextItem(q, "hard", {"answer": a, "difficulty": "hard"})
                for q, a in _arithmetic_hard(rng)]

    if cat == "own_name":
        return [TextItem(_OWN_NAME_PLACEHOLDER, "own_name",
                         {"placeholder": True,
                          "note": "REPLACE THIS: set the participant's own name before "
                                  "running T307/T707; the placeholder string is not a name "
                                  "and must never be presented as one."})]

    if cat == "other_names":
        return [TextItem(n, "other_name", {"role": "filler"}) for n in _OTHER_NAMES]

    if cat == "names_common":
        return [TextItem(n, "name", {"frequency_class": "common_name"}) for n in _COMMON_NAMES]

    raise KeyError(
        f"unknown textbank category {category!r}. Valid categories: "
        f"{sorted(REQUIRED_CATEGORIES + list(ALIASES))}"
    )


def _prose_license(key: str) -> str:
    if key.startswith("aesop"):
        return "public domain (Aesop, traditional; re-told for this repository)"
    if key.startswith("carroll_father"):
        return "public domain (Lewis Carroll, 1855)"
    if key.startswith("carroll_jab"):
        return "public domain (Lewis Carroll, 1871)"
    if key.startswith("lear_"):
        return "public domain (Edward Lear, 1871)"
    return "self-authored (this repository)"


def _passages_for_level(level: str) -> list[str]:
    """Readability ladder: sentence length, clause depth and vocabulary rise.

    * easy  -- one clause per sentence, ~10 words/sentence, concrete nouns.
    * medium -- coordination + one subordinate clause, ~18 words/sentence.
    * hard  -- nominalisations, embedded relative clauses, ~30 words/sentence
      and abstract Latinate vocabulary.

    Mean sentence length is the monotonic variable, because that is what a
    readability score (and a working-memory load) actually tracks; the passages
    are also roughly matched in total length so the ladder is not confounded
    with reading time.
    """
    if level == "easy":
        return [
            "The cat sat on the warm step. The sun was high and the day was hot. "
            "A boy came out of the house. He had a red ball in his hand. He threw the "
            "ball on the grass. The cat did not move. The boy threw the ball again. "
            "This time the cat got up and ran. The boy laughed. Then he sat down on "
            "the step with the cat. The cat came back and sat on his knee. They stayed "
            "there for a long time. It was a good day.",
            "We went to the shop in the morning. The shop was near the park. My mother "
            "wanted bread and milk. I wanted a cake. The man in the shop was kind. He "
            "gave me a small cake for free. We walked home along the river. The water "
            "was grey and slow. A boat went past with two men in it. At home we had tea "
            "and cake by the fire. My father told us a funny story. We all laughed.",
            "It was a cold day. The wind blew hard on the windows. My sister put on her "
            "hat and coat. She went out to feed the birds. The birds came down to the "
            "seed. There were three of them in the snow. She stood very still. Then she "
            "came back in and shut the door. We had soup for lunch. After lunch we played "
            "cards. The fire was warm and the room was quiet.",
        ]
    if level == "medium":
        return [
            "Although the forecast had promised a clear afternoon, dark clouds gathered "
            "over the valley while we were still some distance from the shelter, and we "
            "quickened our pace because the path had been worn away by the winter rain. "
            "The loose stones made every step uncertain, so we kept to the grass at the "
            "edge of the track, where the ground was softer and the walking easier. By "
            "the time the first heavy drops fell, we had reached the old barn, which "
            "leaned against the hillside as though it had given up the struggle to stand "
            "straight, and we waited inside until the worst of the weather had passed.",
            "The committee had considered the proposal for several months before it "
            "finally asked the engineer to prepare a detailed estimate, since the earlier "
            "figures had been based on assumptions that nobody was now willing to defend "
            "in public. The engineer, who had worked on the harbour for more than a "
            "decade, explained that the cost would probably exceed the sum originally "
            "set aside, but he added that a smaller scheme might still be worth building "
            "if the town was prepared to accept a longer delay and a less ambitious "
            "result than the one first described to the residents.",
            "When the library announced that it would close for renovation, the regular "
            "readers, many of whom had used the reading room for years, began to organise "
            "themselves into a small committee that met on Thursday evenings. They "
            "collected signatures, wrote to the council, and proposed that a temporary "
            "room be opened in the community hall, which stood empty for most of the "
            "week. The council agreed in principle, though it warned that the arrangement "
            "would last only until the spring, and that the collection would have to be "
            "reduced while the work was carried out.",
        ]
    # hard
    return [
        "The persistence of the discrepancy between the model's predictions and the "
        "observed measurements suggests that the underlying assumption of stationarity, "
        "which the original formulation inherited from earlier work on homogeneous media "
        "and which subsequent refinements did little to examine, is not merely "
        "inconvenient but fundamentally inapplicable to the conditions under "
        "investigation. Such a conclusion necessitates a substantial reformulation of the "
        "theoretical framework rather than a recalibration of its parameters, and it "
        "implies that the explanatory success attributed to the earlier models depended "
        "on the very regularity whose generality has now been called into question by "
        "evidence that no straightforward modification of the existing apparatus appears "
        "capable of accommodating.",
        "Notwithstanding the considerable methodological sophistication evident in the "
        "commission's final report, the interpretation of its findings remains contested, "
        "insofar as the operationalisation of the central construct presupposes a degree "
        "of conceptual stability that the qualitative evidence assembled during the same "
        "period would appear to undermine. The difficulty is compounded by the "
        "circumstance that the quantitative indicators were selected for their "
        "comparability across jurisdictions rather than for their correspondence to the "
        "theoretical entity whose behaviour the inquiry had been established to "
        "characterise, so that the resulting generalisations, however carefully "
        "qualified, rest on a foundation whose adequacy the report itself leaves "
        "unexamined.",
        "The emergence of distributed systems of production, together with the consequent "
        "reorganisation of labour across national boundaries, has rendered obsolete those "
        "analytical categories that presupposed the coincidence of economic activity "
        "with territorial jurisdiction. The resulting institutional incoherence accounts "
        "for much of the regulatory difficulty now encountered, since the instruments "
        "available to national authorities were designed for an economy whose "
        "characteristic unit was the establishment rather than the network, and the "
        "attempt to extend them to transactions whose coordinates cannot be localised "
        "has produced a proliferation of inconsistent obligations, the cumulative burden "
        "of which falls disproportionately on the smaller participants in the system.",
    ]


def _passages_for_tone(tone: str) -> list[str]:
    """Valence-matched passages: same length band, different emotional content.

    The three versions describe comparable situations (a letter, a walk home, a
    room) with positively valenced, negatively valenced and deliberately flat
    wording, so T506 varies affect and not topic or length.
    """
    if tone == "positive":
        return [
            "The letter arrived on a bright morning and she read it twice at the kitchen "
            "table, laughing out loud the second time. Her oldest friend was coming to "
            "stay, and the house suddenly seemed too small and too quiet to contain the "
            "news. She opened every window, put fresh flowers on the sill, and began to "
            "plan the meals they would cook together, humming while she worked.",
            "The path home ran along the top of the hill, and the evening was so clear "
            "that the whole valley seemed to be holding its breath. He walked slowly, in "
            "no hurry at all, and every gate he passed looked like the entrance to "
            "somewhere he would be welcome. By the time the lights of the town came on "
            "below him, he had decided that the week had been a good one after all.",
        ]
    if tone == "negative":
        return [
            "The letter arrived on a grey morning and she read it twice at the kitchen "
            "table without moving. Her oldest friend was not coming after all, and the "
            "house suddenly seemed too large and too silent to bear the news. She left "
            "every window closed, let the flowers on the sill go dry, and sat for a long "
            "time with her hands folded, listening to the clock in the hall.",
            "The path home ran along the top of the hill, and the evening was so cold that "
            "the whole valley seemed to have withdrawn into itself. He walked quickly, "
            "head down, and every gate he passed looked locked. By the time the lights of "
            "the town came on below him, he had decided that the week had been a waste of "
            "everything he had left.",
        ]
    # neutral
    return [
        "The letter arrived on a Tuesday morning and she read it twice at the kitchen "
        "table. Her oldest friend had sent it from another town, and the house was the "
        "same as it had been on Monday. She opened one window, moved the flowers from the "
        "sill to the counter, and began to plan the rest of the week, writing the items "
        "on a list beside the telephone.",
        "The path home ran along the top of the hill, and the evening was clear enough to "
        "see the road below. He walked at his usual pace, neither fast nor slow, and the "
        "gates he passed were the same gates as always. By the time the lights of the "
        "town came on, he had reached the bottom of the hill and turned towards his "
        "street.",
    ]


# ---------------------------------------------------------------- public API
_BUILT: dict[int, dict[str, list[TextItem]]] = {}


def categories() -> list[str]:
    """Every category name :func:`items` accepts (REQUIRED plus aliases)."""
    return sorted(set(REQUIRED_CATEGORIES) | set(ALIASES))


def build_all(seed: int = DEFAULT_SEED) -> dict[str, list[TextItem]]:
    """Materialise every category (cached per seed in-process; deterministic)."""
    global _BUILT
    if seed not in _BUILT:
        _BUILT[seed] = {cat: _items_for(cat, seed) for cat in REQUIRED_CATEGORIES}
    return _BUILT[seed]


def items(category: str, n: int | None = None, seed: int = DEFAULT_SEED) -> list[TextItem]:
    """Return the items of one category, optionally truncated to ``n``.

    Unknown names raise :class:`KeyError` listing every valid category, and
    close-but-wrong names are resolved through :data:`ALIASES` first: a typo in a
    spec must fail loudly rather than silently yield "no stimuli" and produce an
    empty block.
    """
    key = str(category or "").strip().lower()
    if key in ALIASES:
        key = ALIASES[key]
    if key not in REQUIRED_CATEGORIES:
        close = [c for c in categories() if key and (key in c or c in key)]
        hint = f" Did you mean one of {close}?" if close else ""
        raise KeyError(f"unknown textbank category {category!r}.{hint} "
                       f"Valid categories: {categories()}")
    out = list(build_all(seed).get(key, []))
    if n is not None:
        out = out[: max(0, int(n))]
    return out


def ensure_textbank(force: bool = False) -> Path:
    """Write ``data/stimuli/text/textbank.json`` (idempotent unless ``force``)."""
    ensure_dirs()
    path = Path(TEXTBANK_FILE)
    if path.exists() and not force:
        return path
    bank = build_all(DEFAULT_SEED)
    payload = {
        "version": 1,
        "seed": DEFAULT_SEED,
        "license": ("All items are self-authored for the NeuroBridge repository, except the "
                    "passages marked public domain (Aesop, Carroll 1855/1871, Lear 1871). "
                    "No copyrighted text is included."),
        "aliases": dict(ALIASES),
        "categories": {
            cat: [it.as_dict() for it in its] for cat, its in sorted(bank.items())
        },
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_textbank(path: Path | None = None) -> dict[str, list[TextItem]]:
    """Read back a written bank (used by the report/CLI to show what was shown)."""
    path = Path(path or TEXTBANK_FILE)
    if not path.exists():
        ensure_textbank()
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[TextItem]] = {}
    for cat, its in (data.get("categories") or {}).items():
        out[cat] = [TextItem(text=str(d.get("text", "")), label=str(d.get("label", "")),
                             meta=dict(d.get("meta") or {})) for d in its]
    return out


def count(category: str) -> int:
    """Number of items in a category (without materialising the whole bank)."""
    return len(items(category))


__all__ = [
    "TextItem", "REQUIRED_CATEGORIES", "ALIASES", "TEXTBANK_FILE",
    "COLOURS", "COLOUR_HEX",
    "categories", "items", "pseudowords", "ensure_textbank", "load_textbank",
    "build_all", "count",
]
