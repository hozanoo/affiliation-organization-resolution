from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pycountry
from babel import Locale
from tqdm.auto import tqdm


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LOCATION_OUTPUT_COLUMNS = (
    "address_raw",
    "postcode_raw",
    "city_raw",
    "city_norm",
    "country_raw",
    "country_norm",
)

DIAGNOSTIC_COLUMNS = (
    "country_iso2",
    "country_status",
    "city_status",
    "city_entity_status",
    "city_match_type",
    "city_source_segment",
)

DEFAULT_COUNTRY_ALIASES = {
    "oesterreich": "AT",
    "österreisch": "AT",
    "osterreisch": "AT",
    "bundesrepublik deutschland": "DE",
    "swiss confederation": "CH",
    "schweizerische eidgenossenschaft": "CH",
    "uk": "GB",
    "u.k.": "GB",
    "great britain": "GB",
    "britain": "GB",
    "usa": "US",
    "u.s.a.": "US",
    "us": "US",
    "u.s.": "US",
    "united states of america": "US",
    "united states of america the": "US",
    "pr china": "CN",
    "p.r. china": "CN",
    "prc": "CN",
    "peoples r china": "CN",
    "people's r china": "CN",
    "people s r china": "CN",
    "people's republic of china": "CN",
    "russia": "RU",
    "south korea": "KR",
    "republic of korea": "KR",
    "korea the republic of": "KR",
    "north korea": "KP",
    "iran": "IR",
    "syria": "SY",
    "laos": "LA",
    "moldova": "MD",
    "vietnam": "VN",
    "bolivia": "BO",
    "venezuela": "VE",
    "tanzania": "TZ",
    "brunei": "BN",
    "czech republic": "CZ",
    "cape verde": "CV",
    "ivory coast": "CI",
    "swaziland": "SZ",
    "macedonia": "MK",
    "north macedonia": "MK",
    "turkey": "TR",
    "turkiye": "TR",
    "sudan the": "SD",
    "bahamas the": "BS",
    "gambia the": "GM",
    "netherlands the": "NL",
}

_CITY_COLUMNS = [
    "geoname_id",
    "name",
    "ascii_name",
    "alternate_names",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]

_ALT_COLUMNS = [
    "alternate_name_id",
    "geoname_id",
    "iso_language",
    "alternate_name",
    "is_preferred",
    "is_short",
    "is_colloquial",
    "is_historic",
    "from",
    "to",
]

_LANGUAGE_COLUMNS = [
    "iso_639_3",
    "iso_639_2",
    "iso_639_1",
    "language_name",
]

_STREET_WORDS = (
    r"street|road|avenue|boulevard|lane|drive|"
    r"strasse|straße|gasse|weg|platz|allee|ring|ufer|gürtel|guertel|"
    r"rue|chemin|route|via|viale|piazza|calle|avenida|paseo|"
    r"straat|laan|plein|ulica|aleja|prospekt|cadde|caddesi|sokak"
)

_ORGANIZATION_WORDS = (
    r"university|universität|universitaet|institute|institut|department|"
    r"laboratory|labor|centre|center|company|gmbh|ltd|inc|corporation|"
    r"research|technology|consulting|hospital|academy|school|college|faculty"
)

_HOUSE_NUMBER_RE = re.compile(
    r"(?<!\w)\d{1,6}[A-Za-z]?(?:[/\-]\d+[A-Za-z]?)*(?!\w)"
)

_STREET_RE = re.compile(
    rf"(?:\b(?:{_STREET_WORDS})\b|"
    rf"\b[\wÀ-ÖØ-öø-ÿ]+(?:strasse|straße|gasse|weg|platz)\b)",
    flags=re.IGNORECASE,
)

_ORGANIZATION_RE = re.compile(
    rf"\b(?:{_ORGANIZATION_WORDS})\b",
    flags=re.IGNORECASE,
)

_PO_BOX_RE = re.compile(
    r"\b(?:p\.?\s*o\.?\s*box|postfach|bo[iî]te\s+postale)\b",
    flags=re.IGNORECASE,
)

_POSTCODE_PATTERNS = (
    re.compile(r"^\d{3,7}$"),
    re.compile(r"^[A-Z]{1,3}-?\d{3,6}$", flags=re.IGNORECASE),
    re.compile(r"^\d{5}-\d{4}$"),
    re.compile(r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$", flags=re.IGNORECASE),
    re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$", flags=re.IGNORECASE),
    re.compile(r"^\d{4}\s?[A-Z]{2}$", flags=re.IGNORECASE),
)


def normalize_country_key(value: object) -> str:
    """Normalize a country name or code for exact lookup."""
    if _is_missing(value):
        return ""

    text = str(value).strip().casefold()
    text = re.sub(r"\(\s*the\s*\)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*the\s+", "", text, flags=re.IGNORECASE)
    text = text.replace("’", "'").replace("`", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    if re.fullmatch(r"(?:[a-z]\.){2,4}", text):
        text = text.replace(".", "")
    else:
        text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def build_country_reference(
    locales: Sequence[str] = ("en", "de"),
    additional_aliases: Mapping[str, str] | None = None,
    *,
    include_numeric_codes: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Build an exact country lookup from pycountry, Babel and custom aliases."""
    country_lookup: dict[str, dict[str, Any]] = {}
    ambiguous_aliases: dict[str, set[str]] = {}

    countries_by_iso2 = {
        country.alpha_2: country
        for country in pycountry.countries
    }

    def make_record(country: Any) -> dict[str, Any]:
        return {
            "iso2": country.alpha_2,
            "iso3": country.alpha_3,
            "numeric": getattr(country, "numeric", None),
            "name_en": country.name,
            "official_name": getattr(country, "official_name", None),
            "common_name": getattr(country, "common_name", None),
        }

    def add_alias(alias: object, record: dict[str, Any]) -> None:
        key = normalize_country_key(alias)

        if not key:
            return

        existing = country_lookup.get(key)

        if existing is None:
            country_lookup[key] = record
            return

        if existing["iso2"] == record["iso2"]:
            return

        ambiguous_aliases.setdefault(key, set()).update(
            {existing["iso2"], record["iso2"]}
        )
        country_lookup.pop(key, None)

    for country in pycountry.countries:
        record = make_record(country)

        aliases: set[object] = {
            country.alpha_2,
            country.alpha_3,
            country.name,
            getattr(country, "official_name", None),
            getattr(country, "common_name", None),
        }

        if include_numeric_codes:
            aliases.add(getattr(country, "numeric", None))

        for alias in aliases:
            add_alias(alias, record)

    for locale_name in locales:
        locale = Locale.parse(locale_name)

        for iso2, localized_name in locale.territories.items():
            country = countries_by_iso2.get(iso2)

            if country is not None:
                add_alias(localized_name, make_record(country))

    aliases = dict(DEFAULT_COUNTRY_ALIASES)

    if additional_aliases:
        aliases.update(additional_aliases)

    for alias, iso2 in aliases.items():
        country = countries_by_iso2.get(iso2.upper())

        if country is None:
            raise ValueError(
                f"Invalid ISO-2 code in country aliases: {alias!r} -> {iso2!r}."
            )

        add_alias(alias, make_record(country))

    return country_lookup, ambiguous_aliases


def find_country(
    value: object,
    country_lookup: Mapping[str, dict[str, Any]],
    ambiguous_aliases: Mapping[str, set[str]] | None = None,
) -> dict[str, Any] | None:
    """Find an exact country name or code."""
    key = normalize_country_key(value)

    if not key:
        return None

    if ambiguous_aliases and key in ambiguous_aliases:
        return {
            "matched": False,
            "ambiguous": True,
            "input": value,
            "normalized_input": key,
            "possible_iso2": sorted(ambiguous_aliases[key]),
        }

    record = country_lookup.get(key)

    if record is None:
        return None

    return {
        "matched": True,
        "ambiguous": False,
        "input": value,
        "normalized_input": key,
        **record,
    }


def norm_country(
    value: object,
    resources: Mapping[str, Any],
) -> str | None:
    """Return the normalized English country name for an exact match."""
    result = find_country(
        value,
        resources["country_lookup"],
        resources["ambiguous_country_aliases"],
    )

    if result and result.get("matched"):
        return str(result["name_en"])

    return None


def normalize_city_key(value: object) -> str:
    """Normalize a city name for exact lookup."""
    if _is_missing(value):
        return ""

    text = str(value).strip().casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")

    return re.sub(r"\s+", " ", text).strip()


def build_subdivision_reference() -> dict[str, set[str]]:
    """Build normalized subdivision names and codes grouped by country."""
    lookup: dict[str, set[str]] = {}

    for subdivision in pycountry.subdivisions:
        country_code = str(subdivision.country_code).upper()
        values = {
            normalize_city_key(subdivision.name),
            normalize_city_key(str(subdivision.code).split("-")[-1]),
        }
        lookup.setdefault(country_code, set()).update(
            value for value in values if value
        )

    return lookup


def load_location_resources(
    *,
    data_dir: str | Path | None = None,
    cities_path: str | Path | None = None,
    alternate_names_path: str | Path | None = None,
    language_codes_path: str | Path | None = None,
    locales: Sequence[str] = ("en", "de"),
    additional_country_aliases: Mapping[str, str] | None = None,
    show_progress: bool = True,
    alternate_names_chunksize: int = 1_000_000,
) -> dict[str, Any]:
    """Load country and GeoNames resources used by the location parser."""
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    cities_file = _resolve_data_file(
        cities_path,
        base_dir / "cities500",
        ("cities500.txt", "cities500*"),
        "cities500",
    )
    alternate_names_file = _resolve_data_file(
        alternate_names_path,
        base_dir / "alternateNamesV2",
        ("alternateNamesV2.txt", "alternateNamesV2*"),
        "alternateNamesV2",
    )

    language_file = _resolve_optional_data_file(
        language_codes_path,
        base_dir / "alternateNamesV2",
        ("iso-languagecodes.txt", "iso-languagecodes*"),
    )

    progress = tqdm(
        total=4,
        desc="Loading location data",
        unit="step",
        disable=not show_progress,
    )

    try:
        country_lookup, ambiguous_country_aliases = build_country_reference(
            locales=locales,
            additional_aliases=additional_country_aliases,
        )
        progress.update(1)

        cities_df = load_cities(cities_file)
        progress.update(1)

        alternate_names_df = load_alternate_names(
            alternate_names_file,
            city_ids=cities_df["geoname_id"],
            chunksize=alternate_names_chunksize,
            show_progress=show_progress,
        )
        progress.update(1)

        valid_language_codes = build_language_code_set(language_file)

        city_reference = build_city_reference(
            cities_df,
            alternate_names_df,
            valid_language_codes,
        )
        city_index = build_city_index(city_reference)
        subdivision_lookup = build_subdivision_reference()
        progress.update(1)
    finally:
        progress.close()

    return {
        "country_lookup": country_lookup,
        "ambiguous_country_aliases": ambiguous_country_aliases,
        "city_reference": city_reference,
        "city_index": city_index,
        "subdivision_lookup": subdivision_lookup,
    }


def load_cities(path: str | Path) -> pd.DataFrame:
    """Load a GeoNames cities500 file."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=_CITY_COLUMNS,
        dtype={
            "geoname_id": "Int64",
            "name": "string",
            "ascii_name": "string",
            "alternate_names": "string",
            "feature_class": "string",
            "feature_code": "string",
            "country_code": "string",
            "population": "Int64",
        },
        keep_default_na=True,
        low_memory=False,
    )


def load_alternate_names(
    path: str | Path,
    *,
    city_ids: Iterable[object],
    chunksize: int = 1_000_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load only alternate names that belong to the supplied GeoNames city IDs."""
    ids = {
        int(value)
        for value in city_ids
        if not _is_missing(value)
    }

    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=_ALT_COLUMNS,
        usecols=[
            "geoname_id",
            "iso_language",
            "alternate_name",
            "is_colloquial",
            "is_historic",
        ],
        dtype={
            "geoname_id": "Int64",
            "iso_language": "string",
            "alternate_name": "string",
            "is_colloquial": "string",
            "is_historic": "string",
        },
        keep_default_na=True,
        low_memory=False,
        chunksize=chunksize,
    )

    filtered_chunks: list[pd.DataFrame] = []

    for chunk in tqdm(
        reader,
        desc="Filtering alternate names",
        unit="chunk",
        disable=not show_progress,
        leave=False,
    ):
        filtered = chunk[chunk["geoname_id"].isin(ids)]

        if not filtered.empty:
            filtered_chunks.append(filtered)

    if not filtered_chunks:
        return pd.DataFrame(
            columns=[
                "geoname_id",
                "iso_language",
                "alternate_name",
                "is_colloquial",
                "is_historic",
            ]
        )

    return pd.concat(filtered_chunks, ignore_index=True)


def build_language_code_set(
    language_codes_path: str | Path | None = None,
) -> set[str]:
    """Build the set of language codes accepted for GeoNames alternate names."""
    codes: set[str] = set()

    for language in pycountry.languages:
        for attribute in (
            "alpha_2",
            "alpha_3",
            "bibliographic",
            "terminology",
        ):
            code = getattr(language, attribute, None)

            if code:
                codes.add(str(code).strip().casefold())

    if language_codes_path is None:
        return codes

    languages_df = pd.read_csv(
        language_codes_path,
        sep="\t",
        header=0,
        names=_LANGUAGE_COLUMNS,
        dtype="string",
        keep_default_na=True,
    )

    for column in ("iso_639_3", "iso_639_2", "iso_639_1"):
        codes.update(
            languages_df[column]
            .dropna()
            .str.strip()
            .str.casefold()
            .tolist()
        )

    return codes


def build_city_reference(
    cities_df: pd.DataFrame,
    alternate_names_df: pd.DataFrame,
    valid_language_codes: set[str],
) -> pd.DataFrame:
    """Build the exact city-name reference used for lookup."""
    city_meta = (
        cities_df[
            [
                "geoname_id",
                "name",
                "country_code",
                "population",
                "latitude",
                "longitude",
                "feature_code",
            ]
        ]
        .copy()
        .rename(columns={"name": "canonical_name"})
    )

    canonical_names = cities_df[["geoname_id", "name"]].copy()
    canonical_names = canonical_names.rename(
        columns={"name": "city_name_variant"}
    )
    canonical_names["source"] = "canonical"
    canonical_names["iso_language"] = pd.NA

    ascii_names = cities_df[["geoname_id", "ascii_name"]].copy()
    ascii_names = ascii_names.rename(
        columns={"ascii_name": "city_name_variant"}
    )
    ascii_names["source"] = "ascii"
    ascii_names["iso_language"] = pd.NA

    alternate_names = _filter_safe_alternate_names(
        alternate_names_df,
        valid_language_codes,
    )
    alternate_names = alternate_names[
        ["geoname_id", "alternate_name", "iso_language"]
    ].rename(columns={"alternate_name": "city_name_variant"})
    alternate_names["source"] = "alternate"

    city_names = pd.concat(
        [canonical_names, ascii_names, alternate_names],
        ignore_index=True,
    )

    city_names["geoname_id"] = pd.to_numeric(
        city_names["geoname_id"],
        errors="coerce",
    ).astype("Int64")

    city_names["city_name_variant"] = (
        city_names["city_name_variant"].astype("string").str.strip()
    )
    city_names["city_key"] = (
        city_names["city_name_variant"]
        .map(normalize_city_key)
        .astype("string")
    )

    city_names = city_names[
        city_names["geoname_id"].notna()
        & city_names["city_name_variant"].notna()
        & city_names["city_key"].notna()
        & city_names["city_key"].ne("")
    ].copy()

    source_priority = {
        "canonical": 0,
        "ascii": 1,
        "alternate": 2,
    }

    city_names["_source_priority"] = city_names["source"].map(source_priority)

    city_names = (
        city_names
        .sort_values(["geoname_id", "city_key", "_source_priority"])
        .drop_duplicates(
            subset=["geoname_id", "city_key"],
            keep="first",
        )
        .drop(columns="_source_priority")
    )

    city_reference = city_names.merge(
        city_meta,
        on="geoname_id",
        how="left",
        validate="many_to_one",
    )

    return city_reference.reset_index(drop=True)


def build_city_index(
    city_reference: pd.DataFrame,
) -> dict[str, list[int]]:
    """Build an index from normalized city names to row positions."""
    indices = city_reference.groupby(
        "city_key",
        sort=False,
        observed=True,
    ).indices

    return {
        str(key): positions.tolist()
        for key, positions in indices.items()
    }


def find_city_candidates(
    value: object,
    resources: Mapping[str, Any],
    *,
    country_code: str | None = None,
    max_results: int = 10,
) -> pd.DataFrame:
    """Find exact GeoNames city candidates for a name or alias."""
    city_key = normalize_city_key(value)
    city_reference = resources["city_reference"]
    city_index = resources["city_index"]

    if not city_key or city_key not in city_index:
        return city_reference.iloc[0:0][
            [
                "city_name_variant",
                "canonical_name",
                "country_code",
                "population",
                "geoname_id",
                "source",
            ]
        ].copy()

    result = city_reference.iloc[city_index[city_key]].copy()

    if country_code:
        result = result[
            result["country_code"].str.upper()
            == str(country_code).upper()
        ]

    result = result.drop_duplicates(subset="geoname_id")
    result = result.sort_values(
        "population",
        ascending=False,
        na_position="last",
    )

    return result[
        [
            "city_name_variant",
            "canonical_name",
            "country_code",
            "population",
            "geoname_id",
            "source",
        ]
    ].head(max_results)


def norm_city(
    value: object,
    resources: Mapping[str, Any],
    *,
    country_code: str | None = None,
) -> str | None:
    """Return a canonical city name when all exact matches agree on that name."""
    result = find_city_candidates(
        value,
        resources,
        country_code=country_code,
        max_results=500,
    )

    if result.empty:
        return None

    names = (
        result["canonical_name"]
        .dropna()
        .astype("string")
        .drop_duplicates()
    )

    if len(names) == 1:
        return str(names.iloc[0])

    normalized_names = {
        normalize_city_key(name)
        for name in names
        if normalize_city_key(name)
    }

    if len(normalized_names) == 1:
        return str(names.iloc[0])

    return None


def split_top_level_commas(value: object) -> list[str]:
    """Split on commas outside parentheses."""
    if _is_missing(value):
        return []

    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for character in str(value):
        if character == "(":
            depth += 1
        elif character == ")" and depth > 0:
            depth -= 1

        if character == "," and depth == 0:
            part = "".join(current).strip()

            if part:
                parts.append(part)

            current = []
            continue

        current.append(character)

    final_part = "".join(current).strip()

    if final_part:
        parts.append(final_part)

    return parts


def build_location_candidates(value: object) -> list[dict[str, Any]]:
    """Build exact location candidates from top-level segments and parentheses."""
    if _is_missing(value):
        return []

    text = str(value).strip()
    candidates: list[dict[str, Any]] = []

    for position, part in enumerate(split_top_level_commas(text)):
        candidates.append(
            {
                "source": "comma_segment",
                "position": position,
                "text": part,
            }
        )

    for parenthetical_content in re.findall(r"\(([^()]*)\)", text):
        for position, part in enumerate(parenthetical_content.split(",")):
            part = part.strip()

            if part:
                candidates.append(
                    {
                        "source": "parentheses",
                        "position": position,
                        "text": part,
                    }
                )

    unique_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        key = (
            candidate["source"],
            candidate["text"].casefold(),
        )

        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    return unique_candidates


def remove_trailing_postcode(value: object) -> str:
    """Remove a conservative postcode at the end of a segment."""
    text = _strip_outer_punctuation(value)
    tokens = text.split()

    if not tokens:
        return ""

    if _looks_like_postcode(tokens[-1]):
        tokens.pop()

    return " ".join(tokens).strip()


def city_after_postcode(value: object) -> str:
    """Return the suffix after the last conservative postcode token."""
    text = _strip_outer_punctuation(value)
    tokens = text.split()

    postcode_positions = [
        position
        for position, token in enumerate(tokens)
        if _looks_like_postcode(token)
    ]

    if not postcode_positions:
        return ""

    last_position = postcode_positions[-1]
    city_tokens = tokens[last_position + 1:]

    if not city_tokens or len(city_tokens) > 6:
        return ""

    return " ".join(city_tokens).strip()


def _possible_country_forms(
    value: object,
) -> list[dict[str, str | None]]:
    text = _strip_outer_punctuation(value)

    if not text:
        return []

    forms: list[dict[str, str | None]] = [
        {
            "text": text,
            "postcode": None,
        }
    ]
    tokens = text.split()

    if len(tokens) > 1 and _looks_like_postcode(tokens[0]):
        country_text = " ".join(tokens[1:]).strip()

        if country_text:
            forms.append(
                {
                    "text": country_text,
                    "postcode": tokens[0],
                }
            )

    return forms


def possible_city_forms(value: object) -> list[dict[str, str | None]]:
    """Return conservative city forms derived from one segment."""
    forms: list[dict[str, str | None]] = []

    without_trailing_postcode = remove_trailing_postcode(value)

    if without_trailing_postcode:
        forms.append(
            {
                "text": without_trailing_postcode,
                "match_type": "complete_segment",
                "postcode": _trailing_postcode(value),
            }
        )

    after_postcode = city_after_postcode(value)

    if after_postcode:
        forms.append(
            {
                "text": after_postcode,
                "match_type": "city_after_postcode",
                "postcode": _postcode_before_suffix(value, after_postcode),
            }
        )

    unique_forms: list[dict[str, str | None]] = []
    seen: set[str] = set()

    for form in forms:
        key = normalize_city_key(form["text"])

        if key and key not in seen:
            seen.add(key)
            unique_forms.append(form)

    return unique_forms


def extract_location(
    value: object,
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract conservative country, city, postcode and address information."""
    empty_result = {
        "address_raw": pd.NA,
        "postcode_raw": pd.NA,
        "city_raw": pd.NA,
        "city_norm": pd.NA,
        "country_raw": pd.NA,
        "country_norm": pd.NA,
        "country_iso2": pd.NA,
        "country_status": "not_found",
        "city_status": "not_found",
        "city_entity_status": "not_found",
        "city_match_type": pd.NA,
        "city_source_segment": pd.NA,
        "_country_keys": set(),
        "_country_segments": set(),
        "_region_segments": set(),
        "_city_keys": set(),
        "_address_segments": set(),
        "_postcode_segments": set(),
    }

    if _is_missing(value):
        return empty_result

    candidates = build_location_candidates(value)

    country_hits: list[dict[str, Any]] = []

    for candidate in candidates:
        for country_form in _possible_country_forms(candidate["text"]):
            result = find_country(
                country_form["text"],
                resources["country_lookup"],
                resources["ambiguous_country_aliases"],
            )

            if result and result.get("matched"):
                country_hits.append(
                    {
                        "raw": country_form["text"],
                        "source_segment": candidate["text"],
                        "source": candidate["source"],
                        "postcode": country_form["postcode"],
                        "iso2": result["iso2"],
                        "iso3": result["iso3"],
                        "name": result["name_en"],
                    }
                )
                break

    countries_by_iso2: dict[str, dict[str, Any]] = {}

    for hit in country_hits:
        countries_by_iso2.setdefault(hit["iso2"], hit)

    unique_countries = list(countries_by_iso2.values())

    if len(unique_countries) == 1:
        country = unique_countries[0]
        country_code = country["iso2"]
        country_status = "unique"
    elif len(unique_countries) > 1:
        country = None
        country_code = None
        country_status = "conflict"
    else:
        country = None
        country_code = None
        country_status = "not_found"

    recognized_country_keys = {
        normalize_country_key(hit["raw"])
        for hit in country_hits
    }
    recognized_country_segments = {
        normalize_city_key(hit["source_segment"])
        for hit in country_hits
    }

    top_level_segments = split_top_level_commas(value)
    region_segments: set[str] = set()

    if country_code:
        country_subdivisions = resources.get(
            "subdivision_lookup",
            {},
        ).get(str(country_code).upper(), set())

        for segment in top_level_segments:
            segment_key = normalize_city_key(segment)

            if segment_key in country_subdivisions:
                region_segments.add(segment_key)

    city_hits: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_country_key = normalize_country_key(candidate["text"])
        candidate_segment_key = normalize_city_key(candidate["text"])

        if (
            candidate_country_key in recognized_country_keys
            or candidate_segment_key in recognized_country_segments
            or candidate_segment_key in region_segments
        ):
            continue

        for city_form in possible_city_forms(candidate["text"]):
            searched_city = str(city_form["text"])

            if len(searched_city) < 2:
                continue

            city_results = find_city_candidates(
                searched_city,
                resources,
                country_code=country_code,
                max_results=500,
            ).drop_duplicates(subset="geoname_id")

            if city_results.empty:
                continue

            canonical_names = (
                city_results["canonical_name"]
                .dropna()
                .astype("string")
                .drop_duplicates()
            )
            canonical_keys = {
                normalize_city_key(name)
                for name in canonical_names
                if normalize_city_key(name)
            }

            if len(canonical_keys) == 1:
                city_norm = str(canonical_names.iloc[0])
                canonical_key = next(iter(canonical_keys))
            else:
                city_norm = searched_city
                canonical_key = normalize_city_key(searched_city)

            city_hits.append(
                {
                    "raw": searched_city,
                    "source_segment": candidate["text"],
                    "source": candidate["source"],
                    "match_type": city_form["match_type"],
                    "postcode": city_form["postcode"],
                    "city_norm": city_norm,
                    "canonical_key": canonical_key,
                    "entity_ids": frozenset(
                        int(value)
                        for value in city_results["geoname_id"].dropna()
                    ),
                    "number_of_entities": city_results["geoname_id"].nunique(),
                }
            )

    cities_by_canonical_key: dict[str, dict[str, Any]] = {}

    for hit in city_hits:
        cities_by_canonical_key.setdefault(hit["canonical_key"], hit)

    unique_city_hits = list(cities_by_canonical_key.values())

    if len(unique_city_hits) == 1:
        city = unique_city_hits[0]
        city_status = "recognized"
        city_entity_status = (
            "unique"
            if city["number_of_entities"] == 1
            else "ambiguous"
        )
    elif len(unique_city_hits) > 1:
        city = None
        city_status = "conflict"
        city_entity_status = "conflict"
    else:
        city = None
        city_status = "not_found"
        city_entity_status = "not_found"

    city_keys = set()

    if city is not None:
        city_keys.update(
            {
                normalize_city_key(city["raw"]),
                normalize_city_key(city["city_norm"]),
            }
        )
        city_keys.discard("")

    location_context = (
        country_status == "unique"
        or city_status == "recognized"
    )

    postcode_segments: list[str] = [
        str(hit["postcode"])
        for hit in country_hits
        if hit.get("postcode")
    ]

    for segment in top_level_segments:
        stripped = _strip_outer_punctuation(segment)

        if location_context and _looks_like_postcode(stripped):
            postcode_segments.append(stripped)

    if city is not None and city.get("postcode"):
        postcode_segments.append(str(city["postcode"]))

    address_segments = [
        segment
        for segment in top_level_segments
        if location_context and _looks_like_address(segment)
    ]

    address_raw = _join_unique(address_segments)
    postcode_raw = _join_unique(postcode_segments)

    return {
        "address_raw": address_raw if address_raw else pd.NA,
        "postcode_raw": postcode_raw if postcode_raw else pd.NA,
        "city_raw": city["raw"] if city else pd.NA,
        "city_norm": city["city_norm"] if city else pd.NA,
        "country_raw": country["raw"] if country else pd.NA,
        "country_norm": country["name"] if country else pd.NA,
        "country_iso2": country["iso2"] if country else pd.NA,
        "country_status": country_status,
        "city_status": city_status,
        "city_entity_status": city_entity_status,
        "city_match_type": city["match_type"] if city else pd.NA,
        "city_source_segment": city["source_segment"] if city else pd.NA,
        "_country_keys": recognized_country_keys,
        "_country_segments": recognized_country_segments,
        "_region_segments": region_segments,
        "_city_keys": city_keys,
        "_address_segments": {
            normalize_city_key(segment)
            for segment in address_segments
        },
        "_postcode_segments": {
            normalize_city_key(segment)
            for segment in postcode_segments
        },
    }


def clean_affiliation_location(
    value: object,
    extracted: Mapping[str, Any],
    *,
    remove_addresses: bool = True,
) -> object:
    """Remove only location parts confirmed by the parser."""
    if _is_missing(value):
        return pd.NA

    text = str(value).strip()

    country_keys = set(extracted.get("_country_keys", set()))
    country_segments = set(extracted.get("_country_segments", set()))
    region_segments = set(extracted.get("_region_segments", set()))
    city_keys = set(extracted.get("_city_keys", set()))
    address_segments = set(extracted.get("_address_segments", set()))
    postcode_segments = set(extracted.get("_postcode_segments", set()))

    def clean_parentheses(match: re.Match[str]) -> str:
        content = match.group(1).strip()

        if not content:
            return ""

        parts = [
            part.strip()
            for part in content.split(",")
            if part.strip()
        ]

        if not parts:
            return ""

        if all(
            _is_confirmed_location_part(
                part,
                country_keys=country_keys,
                city_keys=city_keys,
                postcode_segments=postcode_segments,
            )
            for part in parts
        ):
            return ""

        return match.group(0)

    text = re.sub(r"\(([^()]*)\)", clean_parentheses, text)

    cleaned_segments: list[str] = []

    for segment in split_top_level_commas(text):
        stripped = segment.strip()

        if not stripped:
            continue

        normalized_segment = normalize_city_key(stripped)

        if (
            normalize_country_key(stripped) in country_keys
            or normalized_segment in country_segments
            or normalized_segment in region_segments
        ):
            continue

        if _segment_matches_city(stripped, city_keys):
            continue

        if normalized_segment in postcode_segments:
            continue

        if remove_addresses and normalized_segment in address_segments:
            continue

        trimmed = _remove_location_suffix(
            stripped,
            city_keys=city_keys,
        )

        if trimmed != stripped:
            if (
                remove_addresses
                and normalize_city_key(trimmed) in address_segments
            ):
                continue

            if remove_addresses and _looks_like_address(trimmed):
                continue

            if trimmed:
                cleaned_segments.append(trimmed)

            continue

        cleaned_segments.append(stripped)

    cleaned = ", ".join(cleaned_segments)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = cleaned.strip(" ,;:-")

    return cleaned if cleaned else pd.NA


def process_locations(
    df: pd.DataFrame,
    column: str,
    *,
    resources: Mapping[str, Any],
    output_column: str = "affiliation_clean",
    output_columns: Sequence[str] = LOCATION_OUTPUT_COLUMNS,
    remove_addresses: bool = True,
    include_diagnostics: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract location data and create a cleaned affiliation column."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} does not exist.")

    selected_columns = tuple(output_columns)
    invalid_columns = [
        name
        for name in selected_columns
        if name not in LOCATION_OUTPUT_COLUMNS
    ]

    if invalid_columns:
        valid = ", ".join(LOCATION_OUTPUT_COLUMNS)
        invalid = ", ".join(invalid_columns)
        raise ValueError(
            f"Unsupported output column(s): {invalid}. Valid columns: {valid}."
        )

    new_columns = [output_column, *selected_columns]

    if include_diagnostics:
        new_columns.extend(DIAGNOSTIC_COLUMNS)

    existing = [
        name
        for name in new_columns
        if name in df.columns
    ]

    if existing:
        joined = ", ".join(repr(name) for name in existing)
        raise ValueError(f"Output column(s) already exist: {joined}.")

    extracted_rows: list[dict[str, Any]] = []
    cleaned_values: list[object] = []

    values = df[column].tolist()

    for value in tqdm(
        values,
        total=len(values),
        desc="Processing locations",
        unit="row",
        disable=not show_progress,
    ):
        extracted = extract_location(value, resources)
        cleaned = clean_affiliation_location(
            value,
            extracted,
            remove_addresses=remove_addresses,
        )

        extracted_rows.append(extracted)
        cleaned_values.append(cleaned)

    result = df.copy()
    result[output_column] = pd.Series(
        cleaned_values,
        index=result.index,
        dtype="string",
    )

    extracted_frame = pd.DataFrame(extracted_rows, index=result.index)

    for name in selected_columns:
        result[name] = extracted_frame[name]

    if include_diagnostics:
        for name in DIAGNOSTIC_COLUMNS:
            result[name] = extracted_frame[name]

    return result


def _filter_safe_alternate_names(
    alternate_names_df: pd.DataFrame,
    valid_language_codes: set[str],
) -> pd.DataFrame:
    language_value = (
        alternate_names_df["iso_language"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    language_base = language_value.str.split("-", n=1).str[0]

    is_real_language = (
        language_value.isna()
        | language_base.isin(valid_language_codes)
    )
    is_historic = (
        alternate_names_df["is_historic"]
        .fillna("0")
        .eq("1")
    )
    is_colloquial = (
        alternate_names_df["is_colloquial"]
        .fillna("0")
        .eq("1")
    )
    has_name = (
        alternate_names_df["alternate_name"].notna()
        & alternate_names_df["alternate_name"].str.strip().ne("")
    )

    return alternate_names_df[
        is_real_language
        & ~is_historic
        & ~is_colloquial
        & has_name
    ].copy()


def _resolve_data_file(
    path: str | Path | None,
    default_directory: Path,
    patterns: Sequence[str],
    label: str,
) -> Path:
    source = Path(path) if path is not None else default_directory

    if source.is_file():
        return source

    if source.is_dir():
        matches = _matching_files(source, patterns)

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find {label}. Expected a file or directory at {source}."
    )


def _resolve_optional_data_file(
    path: str | Path | None,
    default_directory: Path,
    patterns: Sequence[str],
) -> Path | None:
    if path is not None:
        source = Path(path)

        if source.is_file():
            return source

        if source.is_dir():
            matches = _matching_files(source, patterns)

            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not find the language-code file at {source}."
        )

    if default_directory.is_dir():
        matches = _matching_files(default_directory, patterns)

        if matches:
            return matches[0]

    return None


def _matching_files(
    directory: Path,
    patterns: Sequence[str],
) -> list[Path]:
    matches: list[Path] = []

    for pattern in patterns:
        matches.extend(
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() != ".zip"
        )

    return sorted(set(matches))


def _looks_like_postcode(value: object) -> bool:
    if _is_missing(value):
        return False

    text = _strip_outer_punctuation(value).upper()

    if not text:
        return False

    return any(pattern.fullmatch(text) for pattern in _POSTCODE_PATTERNS)


def _trailing_postcode(value: object) -> str | None:
    tokens = _strip_outer_punctuation(value).split()

    if tokens and _looks_like_postcode(tokens[-1]):
        return tokens[-1]

    return None


def _postcode_before_suffix(
    value: object,
    suffix: str,
) -> str | None:
    text = _strip_outer_punctuation(value)
    suffix_key = normalize_city_key(suffix)
    tokens = text.split()

    for position in range(len(tokens) - 1, -1, -1):
        if not _looks_like_postcode(tokens[position]):
            continue

        candidate_suffix = " ".join(tokens[position + 1:]).strip()

        if normalize_city_key(candidate_suffix) == suffix_key:
            return tokens[position]

    return None


def _looks_like_address(value: object) -> bool:
    if _is_missing(value):
        return False

    text = _strip_outer_punctuation(value)

    if not text:
        return False

    if _PO_BOX_RE.search(text):
        return True

    if _ORGANIZATION_RE.search(text):
        return False

    return bool(
        _STREET_RE.search(text)
        and _HOUSE_NUMBER_RE.search(text)
    )


def _segment_matches_city(
    segment: str,
    city_keys: set[str],
) -> bool:
    if not city_keys:
        return False

    for form in possible_city_forms(segment):
        if (
            form["match_type"] == "complete_segment"
            and normalize_city_key(form["text"]) in city_keys
        ):
            return True

    trimmed = _remove_location_suffix(
        segment,
        city_keys=city_keys,
    )

    return trimmed == ""


def _remove_location_suffix(
    segment: str,
    *,
    city_keys: set[str],
) -> str:
    if not city_keys:
        return segment

    text = _strip_outer_punctuation(segment)
    tokens = text.split()

    postcode_positions = [
        position
        for position, token in enumerate(tokens)
        if _looks_like_postcode(token)
    ]

    for position in reversed(postcode_positions):
        city_part = " ".join(tokens[position + 1:]).strip()

        if normalize_city_key(city_part) not in city_keys:
            continue

        prefix = " ".join(tokens[:position]).strip(" ,;:-")

        return prefix

    return segment


def _is_confirmed_location_part(
    value: str,
    *,
    country_keys: set[str],
    city_keys: set[str],
    postcode_segments: set[str],
) -> bool:
    if normalize_country_key(value) in country_keys:
        return True

    if _segment_matches_city(value, city_keys):
        return True

    if normalize_city_key(value) in postcode_segments:
        return True

    return False


def _join_unique(values: Iterable[object]) -> str:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if _is_missing(value):
            continue

        text = str(value).strip()

        if not text:
            continue

        key = text.casefold()

        if key not in seen:
            seen.add(key)
            result.append(text)

    return "; ".join(result)


def _strip_outer_punctuation(value: object) -> str:
    if _is_missing(value):
        return ""

    return str(value).strip().strip("()[]{}.,;: ")


def _is_missing(value: object) -> bool:
    if value is None:
        return True

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return bool(missing) if isinstance(missing, bool) else False
