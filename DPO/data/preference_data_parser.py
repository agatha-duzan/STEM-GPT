import json
import os
import re
from typing import Literal, TypedDict, cast


__all__ = ("parse_preference_data",)


class SubCriteriaOtherEntry(TypedDict):
    conciseness: Literal["A", "B", "AB", "None"]
    engagement: Literal["A", "B", "AB", "None"]


class CriteriaEntry(TypedDict):
    overall: Literal["A", "B"]
    correctness: Literal["A", "B", "AB", "None"]
    relevance: Literal["A", "B", "AB", "None"]
    clarity: Literal["A", "B", "AB", "None"]
    completeness: Literal["A", "B", "AB", "None"]
    other: SubCriteriaOtherEntry


class SubPreferenceValueEntry(TypedDict):
    A: str
    B: str
    overall: Literal["A", "B"]
    criteria: CriteriaEntry


class PreferenceDataEntry(TypedDict):
    question_id: int
    question_complete: str
    course_id: int
    preference: list[SubPreferenceValueEntry]


def parse_preference_data() -> list[PreferenceDataEntry]:
    #Allowed values for the criteria:
    ALL_VALS = ("A", "B", "AB", "None")
    regular_criteria_pattern = re.compile(r"Conciseness: (A|B|AB|None); Engagement: (A|B|AB|None)")
    #Opening the json file with the preference data and reading it 
    with open(os.path.join(os.path.dirname(__file__), "dpo_preference_data/raw/M1_preference_data_15052024.json")) as f:
        content = json.load(f)
    #Checking that the preference data satisfies a number of constraints. It should be a list of dictionaries:
    if not isinstance(content, list):
        raise ValueError("Expected a list for the json file")
    for entry in content:
        if not isinstance(entry, dict):
            raise ValueError("Entry is not a dictionary")
        if frozenset(entry.keys()) != {"question_id", "question_complete", "course_id", "preference"}:
            raise ValueError("Incorrect key set for an entry")
        all_prefs = entry["preference"]
        if not isinstance(entry["course_id"], int) or not isinstance(entry["question_complete"], str) or not isinstance(entry["course_id"], int) or not isinstance(all_prefs, list):
            raise ValueError("Incorrect typing for the entry values")
        if not isinstance(all_prefs, list):
            raise ValueError("Expected a list for preferences")
        for pref in all_prefs:
            if not isinstance(pref, dict):
                raise ValueError("Preference is not a dictionary")
            if frozenset(pref.keys()) != {"A", "B", "overall", "criteria"}:
                raise ValueError("Incorrect key set for preferences")
            overall_pref = pref["overall"]
            if not isinstance(pref["A"], str) or not isinstance(pref["B"], str) or not isinstance(overall_pref, str) or not isinstance(pref["criteria"], dict):
                raise ValueError("Incorrect values for preferences")
            if overall_pref not in ("A", "B"):
                raise ValueError(f"Preference overall should be 'A' or 'B', not '{overall_pref}'")
            criteria = pref["criteria"]
            if any(not isinstance(x, str) for x in criteria.values()):
                raise ValueError("Incorrect values for criteria")
            if frozenset(criteria.keys()) == {"overall", "correctness", "relevance", "clarity", "completeness", "modified", "other"}:
                del criteria["modified"]
            elif frozenset(criteria.keys()) != {"overall", "correctness", "relevance", "clarity", "completeness", "other"}:
                raise ValueError("Incorrect key set for criteria")
            if criteria["overall"] not in ("A", "B"):
                raise ValueError(f"Overall criteria should be 'A' or 'B', not '{overall_pref}'")
            for key in ("correctness", "relevance", "clarity", "completeness"):
                if criteria[key] not in ALL_VALS:
                    raise ValueError(f"Incorrect value for criteria with key '{key}'")
            if not isinstance(criteria["other"], str):
                raise ValueError("Other pattern is expected to be a string in the original file")
            match = regular_criteria_pattern.fullmatch(criteria["other"])
            if match is None:
                criteria["other"] = {"conciseness": "None", "engagement": "None"}
                continue
            if len(match.groups()) != 2:
                raise ValueError("Incorrect string for 'other'")
            match_groups = match.groups()
            if match_groups[0] not in ALL_VALS or match_groups[1] not in ALL_VALS:
                raise ValueError("Values for 'other' must be in 'A', 'B', 'AB', or 'None'")
            criteria["other"] = {"conciseness": match_groups[0], "engagement": match_groups[1]}

    return cast(list[PreferenceDataEntry], content)
