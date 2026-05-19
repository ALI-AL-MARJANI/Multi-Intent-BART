"""SyntheticOODGenerator: generates out-of-distribution utterances for calibration.

Why we need synthetic OOD data
-------------------------------
Conformal prediction requires a *calibration set* of in-distribution samples
(we use the dev split).  To *evaluate* the OOD detector, we also need OOD
samples — but by definition those don't exist in our dataset.

Solution: generate synthetic OOD utterances with a local LLM via ollama.
We prompt the model to produce short, natural English sentences on topics
that are completely outside the SNIPS / ATIS domains.

SNIPS domains: music, restaurant booking, weather, to-do lists, book search,
               creative work, movie info.
ATIS domains:  airline tickets, flight schedules, airports, ground transport.

Anything outside those domains is OOD: science, sports, cooking, history,
programming, health, etc.

Fallback
--------
If ollama is not installed or the requested model is unavailable, the generator
returns a hardcoded list of OOD utterances.  This keeps the evaluation pipeline
functional even without a local LLM.

Usage
-----
    gen = SyntheticOODGenerator(model="llama3.2:1b")
    utterances = gen.generate(n=200, seed=42)
    gen.save(utterances, "data/ood/snips_ood.txt")
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# ── hardcoded fallback OOD utterances ────────────────────────────────────────
# Topics: science, sports, cooking, history, programming, health, geography.
# None of these overlap with SNIPS/ATIS intents.

_FALLBACK_OOD: list[str] = [
    # science / tech
    "What is the speed of light in a vacuum?",
    "How do black holes form?",
    "Explain quantum entanglement in simple terms.",
    "What is the difference between DNA and RNA?",
    "How does a transistor work?",
    "What causes the northern lights?",
    "How big is the observable universe?",
    "What is the Higgs boson?",
    "How does photosynthesis work?",
    "What is CRISPR gene editing?",
    # sports
    "Who won the last FIFA World Cup?",
    "What are the rules of cricket?",
    "How long is a marathon race?",
    "Who holds the 100m world record?",
    "What is the offside rule in football?",
    "How many players are on a basketball team?",
    "What year did the modern Olympics begin?",
    "How is a tennis tiebreak scored?",
    "What is the Tour de France?",
    "Who has won the most Grand Slam titles?",
    # cooking
    "How do I make a soufflé?",
    "What temperature should I bake bread at?",
    "How do you julienne vegetables?",
    "What is the difference between baking soda and baking powder?",
    "How long should I marinate chicken?",
    "What herbs go well with lamb?",
    "How do I make homemade pasta from scratch?",
    "What is the Maillard reaction in cooking?",
    "How do I caramelize onions properly?",
    "What is the difference between stock and broth?",
    # history
    "When did the French Revolution begin?",
    "Who was the first Roman emperor?",
    "What caused World War One?",
    "When was the printing press invented?",
    "Who built the Great Wall of China?",
    "What was the Byzantine Empire?",
    "When did the Berlin Wall fall?",
    "Who was Cleopatra?",
    "What happened during the Black Death?",
    "Who discovered America?",
    # programming / tech
    "What is the difference between Python and Java?",
    "How does a hash table work?",
    "What is recursion in programming?",
    "How does garbage collection work?",
    "What is a REST API?",
    "What is the difference between HTTP and HTTPS?",
    "How does blockchain technology work?",
    "What is machine learning?",
    "What is the difference between RAM and ROM?",
    "How does a compiler work?",
    # health / biology
    "What is the difference between a virus and a bacterium?",
    "How does the immune system work?",
    "What is cholesterol and why does it matter?",
    "How much sleep does an adult need?",
    "What is the function of the pancreas?",
    "How does blood pressure get measured?",
    "What is a calorie?",
    "How do vaccines work?",
    "What is the difference between type 1 and type 2 diabetes?",
    "What causes allergies?",
    # geography / nature
    "What is the longest river in the world?",
    "How are mountains formed?",
    "What is the difference between a hurricane and a typhoon?",
    "How deep is the Mariana Trench?",
    "What causes earthquakes?",
    "What is the largest country by area?",
    "How do volcanoes erupt?",
    "What is the Amazon rainforest?",
    "How many oceans are there?",
    "What causes tides?",
    # math / logic
    "What is Euler's number?",
    "How do you calculate the area of a circle?",
    "What is the Pythagorean theorem?",
    "What is a prime number?",
    "What is the Fibonacci sequence?",
    "How does binary counting work?",
    "What is a derivative in calculus?",
    "What is the difference between permutations and combinations?",
    "What is pi and where does it come from?",
    "How do you solve a quadratic equation?",
    # finance / economy
    "What is inflation?",
    "How does the stock market work?",
    "What is a mortgage?",
    "What is the difference between a recession and a depression?",
    "How does compound interest work?",
    "What is GDP?",
    "What is cryptocurrency?",
    "How are exchange rates determined?",
    "What is a hedge fund?",
    "What causes inflation?",
]


class SyntheticOODGenerator:
    """Generates OOD utterances using a local LLM via ollama (with fallback).

    Args:
        model:      ollama model name (e.g. "llama3.2:1b", "mistral:7b").
        in_domain:  short description of in-domain topics to explicitly exclude.
    """

    _IN_DOMAIN_SNIPS = (
        "music playback, restaurant booking, weather queries, to-do lists, "
        "book search, creative work requests, movie information"
    )
    _IN_DOMAIN_ATIS = (
        "airline tickets, flight schedules, airports, ground transport, "
        "flight times, airline codes, aircraft types"
    )

    def __init__(
        self,
        model: str = "llama3.2:1b",
        in_domain: str | None = None,
    ) -> None:
        self.model = model
        self.in_domain = in_domain or f"{self._IN_DOMAIN_SNIPS} and {self._IN_DOMAIN_ATIS}"
        self._ollama_available = self._check_ollama()

    # ── public API ─────────────────────────────────────────────────────────────

    def generate(self, n: int = 200, seed: int = 42) -> list[str]:
        """Generate n OOD utterances.

        Tries ollama first; falls back to the hardcoded list if unavailable.
        """
        rng = random.Random(seed)

        if self._ollama_available:
            utterances = self._generate_with_ollama(n, rng)
            if utterances:
                logger.info("Generated %d OOD utterances via ollama (%s)", len(utterances), self.model)
                return utterances
            logger.warning("ollama generation failed — falling back to hardcoded OOD list.")

        # fallback: sample from hardcoded list with replacement if needed
        if n <= len(_FALLBACK_OOD):
            return rng.sample(_FALLBACK_OOD, n)
        # sample with replacement for n > len(fallback)
        return [rng.choice(_FALLBACK_OOD) for _ in range(n)]

    @staticmethod
    def save(utterances: list[str], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for utt in utterances:
                f.write(utt.strip() + "\n")
        logger.info("Saved %d OOD utterances to %s", len(utterances), path)

    @staticmethod
    def load(path: str | Path) -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # ── internals ──────────────────────────────────────────────────────────────

    def _check_ollama(self) -> bool:
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            logger.info(
                "ollama Python package not found. "
                "Install with: pip install ollama  and  ollama pull %s",
                self.model,
            )
            return False

    def _generate_with_ollama(self, n: int, rng: random.Random) -> list[str]:
        """Use ollama to generate n OOD utterances in batches of 20."""
        try:
            import ollama

            topics = [
                "science and physics", "sports and athletics", "cooking and recipes",
                "history and ancient civilizations", "programming and software",
                "human biology and health", "geography and nature",
                "mathematics and logic", "finance and economics",
                "art and literature", "astronomy and space", "psychology",
            ]

            utterances: list[str] = []
            batch_size = 20

            while len(utterances) < n:
                topic = rng.choice(topics)
                prompt = (
                    f"Generate {batch_size} short, natural English questions or requests "
                    f"about {topic}. "
                    f"These must NOT be about: {self.in_domain}. "
                    f"Return ONLY a JSON array of strings, no explanation. "
                    f"Example format: [\"question 1\", \"question 2\", ...]"
                )

                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.9, "num_predict": 512},
                )
                content = response["message"]["content"].strip()

                # extract JSON array from response
                start = content.find("[")
                end = content.rfind("]") + 1
                if start == -1 or end == 0:
                    continue
                try:
                    batch = json.loads(content[start:end])
                    utterances.extend(str(u).strip() for u in batch if u)
                except json.JSONDecodeError:
                    continue

            return utterances[:n]

        except Exception as e:  # noqa: BLE001
            logger.warning("ollama generation raised: %s", e)
            return []
