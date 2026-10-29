"""experience: rookie = first NBA season this one, matched across DK's and nba_api's spellings."""
from ivan.sports.nba import experience as X


EXP = dict(first_year={X.loose("Robert Williams III"): 2018, X.loose("David Jones Garcia"): 2025, X.loose("Elijah Harkless"): 2024,
                       X.loose("Darius Acuff Jr."): 2026, X.loose("Isaac Jones"): 2024},
           team={X.loose("Robert Williams III"): "POR", X.loose("David Jones Garcia"): "SAS", X.loose("Elijah Harkless"): "DET",
                 X.loose("Darius Acuff Jr."): "DET", X.loose("Isaac Jones"): "DET"})


def test_loose_names_meet():
    assert X.loose("Robert Williams III") == X.loose("robert williams")
    assert X.loose("David Jones Garcia") == X.loose("david jonesgarcia")
    assert X.loose("Darius Acuff Jr.") == X.loose("darius acuff jr")


def test_is_rookie():
    assert X.is_rookie("darius acuff jr", 2026, EXP, played_before=False)
    assert not X.is_rookie("robert williams", 2026, EXP, played_before=False)       # missed last season: still a veteran
    assert not X.is_rookie("david jonesgarcia", 2026, EXP, played_before=False)
    assert not X.is_rookie("ej harkless", 2026, EXP, played_before=False, team="DET")   # nickname: the one Harkless on DET
    assert X.is_rookie("ej harkless", 2026, EXP, played_before=False, team="BOS")       # not found: the box-score rule
    assert X.is_rookie("ak okereke", 2026, EXP, played_before=False)
    assert not X.is_rookie("ak okereke", 2026, EXP, played_before=True)
