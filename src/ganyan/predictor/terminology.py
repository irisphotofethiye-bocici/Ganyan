"""TJK-aligned display names for bet strategies and bet-slip terms.

Keep internal strategy identifiers (uclu_top1 etc.) stable for DB, but map to
official TJK e-bayi terminology for CLI and web output so the user isn't
translating in their head.

Learned 2026-04-23:
  - "K" toggle on the TJK bet slip is Komple, NOT Kutu.
  - Kutu is auto-applied when same 3 horses selected in all 3 columns.
  - Sıralı Üçlü tek kombinasyon has 20 TL minimum ticket.
  - "Bahis sayısı" means different things pre-bet vs post-bet screens.
"""

from __future__ import annotations

STRATEGY_DISPLAY_TR: dict[str, str] = {
    "uclu_top1": "Sıralı Üçlü Bahis (tek kombinasyon)",
    "uclu_box6": "Sıralı Üçlü Bahis (Kutu 6)",
    "sirali_ikili_top1": "Sıralı İkili Bahis (tek kombinasyon)",
    "ganyan_top1": "Ganyan (referans)",
    "plase_top1": "Plase (banko, top-2)",
}

STRATEGY_DISPLAY_SHORT_TR: dict[str, str] = {
    "uclu_top1": "Üçlü Tek",
    "uclu_box6": "Üçlü Kutu 6",
    "sirali_ikili_top1": "İkili Sıralı Tek",
    "ganyan_top1": "Ganyan",
    "plase_top1": "Plase",
}

STRATEGY_BET_SLIP_STEPS_TR: dict[str, list[str]] = {
    "uclu_top1": [
        "Bahis türü: Sıralı Üçlü Bahis",
        "1.At sütunu: sadece 1. sıra atı",
        "2.At sütunu: sadece 2. sıra atı",
        "3.At sütunu: sadece 3. sıra atı",
        "Bahis sayısı: 1 (pre-bet) / misli × 1 (post-bet)",
        "⚠️ Minimum 20 TL",
    ],
    "uclu_box6": [
        "Bahis türü: Sıralı Üçlü Bahis",
        "Her 3 sütunda aynı 3 atı işaretle (sistem Kutu olarak otomatik filtreler)",
        "Komple/K toggle: KAPALI (dokunma)",
        "Bahis sayısı: 6 (pre-bet) / 6 × misli (post-bet)",
        "Minimum ~12 TL",
    ],
    "sirali_ikili_top1": [
        "Bahis türü: Sıralı İkili Bahis",
        "1.At: sadece 1. sıra atı, 2.At: sadece 2. sıra atı",
        "Bahis sayısı: 1 (pre-bet)",
    ],
    "ganyan_top1": [
        "Bahis türü: Ganyan",
        "Tek at seç",
        "Minimum 1 TL",
    ],
}

TJK_POOL_CHEATSHEET_TR: list[tuple[str, str]] = [
    ("Ganyan", "Tek at, 1. sırayı alır."),
    ("Plase", "Tek at, ilk 3'e girer."),
    ("İkili Bahis", "İki at, ilk 2 (sıra önemsiz)."),
    ("Sıralı İkili Bahis", "İki at, tam sırayla 1-2."),
    ("Sıralı Üçlü Bahis", "Üç at, tam sırayla 1-2-3. Kutu için aynı 3 atı 3 sütuna seç."),
    ("Sıralı Beşli Bahis", "Beş at, tam sırayla 1-2-3-4-5 (çok düşük isabet)."),
    ("Çifte Bahis", "2 ardışık yarışın kazananı."),
    ("3'lü / 4'lü / 6'lı / 7'li Ganyan", "N ardışık yarışın kazananı — BİZİM STRATEJİMİZ DEĞİL."),
    ("Tabela Bahis", "Sabit oranlı bahis (müşterek değil)."),
]

BET_SLIP_WIDGET_NOTES_TR: list[tuple[str, str]] = [
    ("K toggle (yeşil)", "Komple modu — sahadaki tüm atları o sütuna ekler. ⚠️ Kutu DEĞİL."),
    ("Kutu", "Ayrı bir buton yok. Aynı 3 atı 3 sütuna seçince sistem otomatik 6 permütasyona indirger."),
    ("Virgüllü", "Bahis türü dropdown'ındaki alternatif giriş formatı. Aynı havuz, farklı arayüz."),
    ("Misli", "Kombinasyon başına stake çarpanı. Stake = kombinasyon × birim × misli."),
    ("Bahis sayısı", "Pre-bet = benzersiz kombinasyon; post-bet = kombinasyon × misli (karıştırma!)."),
    ("Tutar", "Toplam bilet maliyeti. Birim fiyatı Sıralı Üçlü için 2 TL."),
]

MINIMUM_STAKES_TL: dict[str, float] = {
    "uclu_top1": 20.0,
    "uclu_box6": 12.0,
    "sirali_ikili_top1": 2.0,
    "ganyan_top1": 1.0,
    "plase": 1.0,
    "plase_top1": 1.0,
}


def strategy_display(strategy: str, short: bool = False) -> str:
    """Return TJK-aligned Turkish display name for a strategy."""
    table = STRATEGY_DISPLAY_SHORT_TR if short else STRATEGY_DISPLAY_TR
    return table.get(strategy, strategy)


def bet_slip_steps(strategy: str) -> list[str]:
    """Return step-by-step TJK bet-slip instructions for a strategy."""
    return STRATEGY_BET_SLIP_STEPS_TR.get(strategy, [])


def min_stake_tl(strategy: str) -> float:
    """Return the TJK minimum ticket value for a strategy, in TL."""
    return MINIMUM_STAKES_TL.get(strategy, 0.0)
