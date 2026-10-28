"""Word-relation pairs used in AMAT5315 Week 8, Part 1.

Each relation contains exactly nine ordered (A, B) pairs.  The analysis
compares the embedding offset ``unit(B) - unit(A)`` with the mean offset of
the other eight pairs (leave-one-out).
"""

from __future__ import annotations

RELATIONS: dict[str, list[tuple[str, str]]] = {
    "country_capital": [
        ("iran", "tehran"),
        ("turkey", "ankara"),
        ("russia", "moscow"),
        ("china", "beijing"),
        ("france", "paris"),
        ("sweden", "stockholm"),
        ("japan", "tokyo"),
        ("peru", "lima"),
        ("kenya", "nairobi"),
    ],
    "male_female": [
        ("king", "queen"),
        ("man", "woman"),
        ("he", "she"),
        ("brother", "sister"),
        ("father", "mother"),
        ("son", "daughter"),
        ("uncle", "aunt"),
        ("boy", "girl"),
        ("husband", "wife"),
    ],
    "adjective_comparative": [
        ("big", "bigger"),
        ("small", "smaller"),
        ("fast", "faster"),
        ("dark", "darker"),
        ("cold", "colder"),
        ("old", "older"),
        ("strong", "stronger"),
        ("long", "longer"),
        ("tall", "taller"),
    ],
}

RELATION_TITLES: dict[str, str] = {
    "country_capital": "Country -> capital",
    "male_female": "Male -> female",
    "adjective_comparative": "Adjective -> comparative",
}
