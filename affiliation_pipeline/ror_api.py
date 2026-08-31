from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import requests
from tqdm.auto import tqdm


@dataclass(frozen=True)
class RORMatch:
    """Internal representation of a ROR matching decision."""

    status: str
    status_detail: str
    ror_id: str = ""
    display_name: str = ""
    ror_status: str = ""
    matched_name: str = ""
    matched_name_types: tuple[str, ...] = ()
    matched_candidate: str = ""
    matching_type: str = ""
    successor_id: str = ""
    successor_name: str = ""
    record_names: tuple[tuple[str, tuple[str, ...]], ...] = ()
    country_codes: tuple[str, ...] = ()
    country_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExactCandidate:
    """Organization candidate used by the exact-name fallback."""

    text: str
    score: int
    starts_with_subunit: bool
    is_acronym: bool


@dataclass(frozen=True)
class OrganizationSpan:
    """Position of the matched organization inside affiliation_clean."""

    start: int
    end: int
    matched_text: str
    prefix_remainder: str = ""
    suffix_remainder: str = ""
    score: float = 0.0


class RORAffiliationProcessor:
    """Match prepared affiliation data to ROR and structure clean columns."""

    VERSION = "4.2.0"
    MATCHING_VERSION = "4.2"

    VALID_STATUSES = {
        "pending",
        "matched",
        "unmatched",
        "review",
        "api_error",
        "rate_limited",
        "skipped",
    }

    DEFAULT_RETRY_STATUSES = (
        "pending",
        "api_error",
        "rate_limited",
    )

    STRONG_ORG_RE = re.compile(
        r"\b("
        r"university|universität|universite|université|college|"
        r"academy|agency|association|authority|commission|company|"
        r"corporation|foundation|hospital|institute|institut|museum|"
        r"observatory|research centre|research center|society|"
        r"administration|gmbh|ag|ltd\.?|limited|inc\.?|corp\.?"
        r")\b",
        re.IGNORECASE,
    )

    SUBUNIT_START_RE = re.compile(
        r"^\s*("
        r"department|dept\.?|faculty|school|graduate school|"
        r"laboratory|laboratories|lab\.?|unit|section|group|team|"
        r"chair|division|centre|center|clinic|"
        r"abteilung|fakultät|lehrstuhl|département|faculté"
        r")\b",
        re.IGNORECASE,
    )

    NON_ORG_START_RE = re.compile(
        r"^\s*("
        r"room|building|floor|postal|postcode|zip|"
        r"p\.?\s*o\.?\s*box|postfach"
        r")\b",
        re.IGNORECASE,
    )

    ROR_ID_RE = re.compile(
        r"(?:https?://)?(?:www\.)?ror\.org/"
        r"(0[a-hj-km-np-tv-z0-9]{6}[0-9]{2})",
        re.IGNORECASE,
    )
    ROR_ID_BARE_RE = re.compile(
        r"^(0[a-hj-km-np-tv-z0-9]{6}[0-9]{2})$",
        re.IGNORECASE,
    )

    GENERIC_TOKENS = {
        "university",
        "universitat",
        "universite",
        "institute",
        "institut",
        "college",
        "academy",
        "research",
        "center",
        "centre",
        "department",
        "faculty",
        "school",
        "hospital",
        "foundation",
        "association",
        "society",
        "company",
        "corporation",
        "limited",
        "gmbh",
    }

    def __init__(
        self,
        *,
        api_input_col: str = "affiliation_part_raw",
        clean_col: str = "affiliation_clean",
        original_ror_col: str | None = None,
        country_code_col: str | None = "country_iso2",
        country_name_col: str | None = "country_norm",
        clean_output_prefix: str = "affiliation_clean",
        output_ror_col: str = "ror_after_api",
        ror_status_col: str = "ror_status",
        status_col: str = "status",
        status_detail_col: str = "status_detail",
        successor_id_col: str = "ror_successor_id",
        successor_name_col: str = "ror_successor_name",
        cache_file: str | Path | None = None,
        env_file: str | Path | None = None,
        client_id: str | None = None,
        max_fallback_queries: int = 3,
        timeout: int = 30,
        max_retries: int = 4,
        min_request_interval: float | None = None,
        checkpoint_every: int = 100,
        chosen_locator_threshold: float = 0.72,
    ) -> None:
        self.api_input_col = api_input_col
        self.clean_col = clean_col
        self.original_ror_col = original_ror_col
        self.country_code_col = country_code_col
        self.country_name_col = country_name_col

        self.clean_output_prefix = clean_output_prefix
        self.output_ror_col = output_ror_col
        self.ror_status_col = ror_status_col
        self.status_col = status_col
        self.status_detail_col = status_detail_col
        self.successor_id_col = successor_id_col
        self.successor_name_col = successor_name_col

        project_root = Path(__file__).resolve().parents[1]

        self.cache_file = (
            Path(cache_file)
            if cache_file is not None
            else project_root / "data" / "ror_api_cache.json"
        )
        self.env_file = (
            Path(env_file)
            if env_file is not None
            else project_root / ".env"
        )

        self.client_id = (
            str(client_id).strip()
            if client_id is not None
            else self._load_client_id()
        )

        self.max_fallback_queries = int(max_fallback_queries)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.checkpoint_every = int(checkpoint_every)
        self.chosen_locator_threshold = float(chosen_locator_threshold)

        if self.max_fallback_queries < 1:
            raise ValueError("max_fallback_queries must be at least 1.")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be greater than 0.")
        if not 0 <= self.chosen_locator_threshold <= 1:
            raise ValueError(
                "chosen_locator_threshold must be between 0 and 1."
            )

        if min_request_interval is None:
            self.min_request_interval = 0.25 if self.client_id else 6.5
        else:
            self.min_request_interval = float(min_request_interval)

        if self.min_request_interval < 0:
            raise ValueError("min_request_interval must not be negative.")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": f"AffiliationPipelineROR/{self.VERSION}",
            }
        )

        if self.client_id:
            self.session.headers["Client-Id"] = self.client_id

        self._last_request_at = 0.0
        self.cache = self._load_cache()
        self.output_columns: list[str] = []

    def process(
        self,
        df: pd.DataFrame,
        *,
        statuses: Sequence[str] | None = None,
        max_rows: int | None = None,
        force: bool = False,
        bypass_cache: bool = False,
        checkpoint_file: str | Path | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Process selected rows and return a new DataFrame."""
        result = df.copy()

        self._validate_input_columns(result)
        self.output_columns = self._prepare_output_columns(result)

        selected = self._select_rows(
            result,
            statuses=statuses,
            max_rows=max_rows,
            force=force,
        )

        if selected.empty:
            return result

        checkpoint_path = (
            Path(checkpoint_file)
            if checkpoint_file is not None
            else None
        )

        for processed, index in enumerate(
            tqdm(
                selected.index,
                total=len(selected),
                desc="Matching ROR",
                unit="row",
                disable=not show_progress,
            ),
            start=1,
        ):
            self._process_row(
                result,
                index,
                bypass_cache=bypass_cache,
            )

            if (
                processed % self.checkpoint_every == 0
                or processed == len(selected)
            ):
                self._atomic_save_cache()

                if checkpoint_path is not None:
                    self._atomic_save_dataframe(
                        result,
                        checkpoint_path,
                    )

        return result

    def generate_exact_candidates(
        self,
        value: object,
        *,
        limit: int | None = None,
    ) -> list[str]:
        """Return ranked candidates without calling ROR."""
        candidates = self._generate_exact_candidates(value)

        if limit is not None:
            candidates = candidates[: int(limit)]

        return [candidate.text for candidate in candidates]

    def _process_row(
        self,
        df: pd.DataFrame,
        index: Any,
        *,
        bypass_cache: bool,
    ) -> None:
        row = df.loc[index]
        api_input = self._safe_text(row.get(self.api_input_col))
        clean_input = self._safe_text(row.get(self.clean_col))

        self._write_initial_split(df, index, clean_input)
        self._clear_metadata(df, index)

        if not api_input:
            self._write_metadata(
                df,
                index,
                RORMatch(
                    status="skipped",
                    status_detail="empty_api_input",
                ),
            )
            return

        chosen = self._match_affiliation(
            api_input,
            bypass_cache=bypass_cache,
        )

        chosen_mismatch: RORMatch | None = None

        if chosen.status == "matched":
            # Validate ROR's chosen result against the original API input,
            # not against affiliation_clean, because location cleaning may
            # intentionally remove country qualifiers that are part of a ROR name.
            raw_span = self._find_organization_span(
                api_input,
                chosen,
            )

            if (
                raw_span is None
                or raw_span.score < self.chosen_locator_threshold
            ):
                chosen_mismatch = RORMatch(
                    status="review",
                    status_detail="chosen_name_mismatch",
                )
                chosen = RORMatch(
                    status="unmatched",
                    status_detail="chosen_name_mismatch",
                )

        if chosen.status == "unmatched":
            match = self._match_exact_fallback(
                clean_input or api_input,
                row=row,
                bypass_cache=bypass_cache,
            )

            if (
                match.status == "unmatched"
                and chosen_mismatch is not None
            ):
                match = chosen_mismatch
        else:
            match = chosen

        original_ror = (
            self._safe_text(row.get(self.original_ror_col))
            if self.original_ror_col is not None
            else ""
        )

        if original_ror:
            if match.ror_id:
                match = self._compare_with_original_ror(
                    match,
                    original_ror,
                )
            elif match.status in {"unmatched", "review"}:
                original_match = self._validate_original_ror(
                    original_ror,
                    clean_input or api_input,
                    row=row,
                    bypass_cache=bypass_cache,
                )

                if original_match.ror_id:
                    match = original_match

        span: OrganizationSpan | None = None

        if match.ror_id:
            span = self._find_organization_span(
                clean_input,
                match,
                preferred_candidate=match.matched_candidate,
            )

            # A safe ROR match plus a single remaining clean segment is enough
            # to normalize affiliation_clean_1 even when a country qualifier
            # removed by locations.py prevents exact local name localization.
            if span is None:
                clean_parts = self._split_top_level_commas(clean_input)

                if len(clean_parts) == 1:
                    span = OrganizationSpan(
                        start=0,
                        end=0,
                        matched_text=clean_parts[0],
                        score=1.0,
                    )

        if self._may_structure(match) and span is not None:
            remaining = self._remaining_after_span(
                self._split_top_level_commas(clean_input),
                span,
            )
            remaining = self._remove_equivalent_record_names(
                remaining,
                match,
            )

            if self._suspect_multiple_organizations(remaining):
                match = replace(
                    match,
                    status="review",
                    status_detail="multiple_organizations",
                )
            else:
                self._write_structured_match(
                    df,
                    index,
                    match,
                    span,
                    remaining,
                )

        self._write_metadata(df, index, match)

    def _match_affiliation(
        self,
        text: str,
        *,
        bypass_cache: bool,
    ) -> RORMatch:
        cache_key = self._cache_key("affiliation", text)

        if not bypass_cache:
            cached = self.cache.get(cache_key)

            if cached is not None:
                return self._match_from_cache(cached)

        payload = self._request(
            params={"affiliation": text},
        )

        if isinstance(payload, RORMatch):
            return payload

        chosen = next(
            (
                item
                for item in payload.get("items", [])
                if item.get("chosen") is True
            ),
            None,
        )

        if chosen is None:
            result = RORMatch(
                status="unmatched",
                status_detail="no_chosen_result",
            )
        else:
            organization = chosen.get("organization", {}) or {}

            result = self._match_from_record(
                organization,
                status="matched",
                status_detail="affiliation_chosen",
                matching_type=self._safe_text(
                    chosen.get("matching_type")
                ),
            )

            if not result.ror_id or not result.display_name:
                result = RORMatch(
                    status="api_error",
                    status_detail="incomplete_chosen_record",
                )

        if result.status in {"matched", "unmatched", "review"}:
            self.cache[cache_key] = self._match_to_cache(result)

        return result

    def _match_exact_fallback(
        self,
        value: str,
        *,
        row: pd.Series,
        bypass_cache: bool,
    ) -> RORMatch:
        candidates = self._generate_exact_candidates(value)

        if not candidates:
            return RORMatch(
                status="unmatched",
                status_detail="no_exact_candidate",
            )

        resolved: list[tuple[ExactCandidate, RORMatch]] = []
        review_results: list[RORMatch] = []

        for candidate in candidates[: self.max_fallback_queries]:
            result = self._query_exact_candidate(
                candidate,
                row=row,
                bypass_cache=bypass_cache,
            )

            if result.status in {"api_error", "rate_limited"}:
                return result

            if result.ror_id:
                resolved.append((candidate, result))
            elif result.status == "review":
                review_results.append(result)

        distinct_ids = {
            result.ror_id
            for _, result in resolved
            if result.ror_id
        }

        if len(distinct_ids) > 1:
            return RORMatch(
                status="review",
                status_detail="multiple_exact_candidates",
            )

        if len(distinct_ids) == 1:
            _, best = max(
                resolved,
                key=lambda item: item[0].score,
            )
            return best

        if review_results:
            priorities = {
                "ambiguous_exact_name": 4,
                "alternative_name_without_country": 2,
                "acronym_exact_without_country": 2,
            }

            return max(
                review_results,
                key=lambda item: priorities.get(
                    item.status_detail,
                    0,
                ),
            )

        return RORMatch(
            status="unmatched",
            status_detail="no_exact_name_match",
        )

    def _query_exact_candidate(
        self,
        candidate: ExactCandidate,
        *,
        row: pd.Series,
        bypass_cache: bool,
    ) -> RORMatch:
        country_context = self._country_cache_context(row)

        cache_key = self._cache_key(
            "exact",
            f"{country_context}|{candidate.text}",
        )

        if not bypass_cache:
            cached = self.cache.get(cache_key)

            if cached is not None:
                return self._match_from_cache(cached)

        payload = self._request(
            params={
                "query": self._quoted_query(candidate.text),
                "all_status": "",
            }
        )

        if isinstance(payload, RORMatch):
            return payload

        exact_records: list[
            tuple[dict[str, Any], str, tuple[str, ...]]
        ] = []

        for organization in payload.get("items", []) or []:
            exact_name = self._find_exact_record_name(
                candidate.text,
                organization,
            )

            if exact_name is None:
                continue

            matched_name, matched_types = exact_name
            exact_records.append(
                (
                    organization,
                    matched_name,
                    matched_types,
                )
            )

        if not exact_records:
            result = RORMatch(
                status="unmatched",
                status_detail="no_exact_name_match",
            )
            self.cache[cache_key] = self._match_to_cache(result)
            return result

        filtered, country_used, country_conflict = (
            self._filter_records_by_country(
                exact_records,
                row,
            )
        )

        if country_conflict:
            result = RORMatch(
                status="unmatched",
                status_detail="exact_name_country_conflict",
            )
            self.cache[cache_key] = self._match_to_cache(result)
            return result

        considered = filtered if country_used else exact_records

        unique_records: dict[
            str,
            tuple[dict[str, Any], str, tuple[str, ...]],
        ] = {}

        for record in considered:
            ror_id = self._safe_text(record[0].get("id"))

            if ror_id:
                unique_records.setdefault(ror_id, record)

        if len(unique_records) != 1:
            result = RORMatch(
                status="review",
                status_detail="ambiguous_exact_name",
            )
            self.cache[cache_key] = self._match_to_cache(result)
            return result

        organization, matched_name, matched_types = next(
            iter(unique_records.values())
        )

        record_status = self._safe_text(
            organization.get("status")
        ).casefold()

        if record_status == "inactive":
            status = "review"
            detail = "exact_inactive_match"
        elif record_status == "withdrawn":
            status = "review"
            detail = "exact_withdrawn_match"
        elif "acronym" in matched_types and not country_used:
            status = "review"
            detail = "acronym_exact_without_country"
        elif (
            "ror_display" not in matched_types
            and not country_used
        ):
            status = "review"
            detail = "alternative_name_without_country"
        else:
            status = "matched"

            if "ror_display" in matched_types:
                detail = (
                    "exact_name_country_fallback"
                    if country_used
                    else "exact_name_fallback"
                )
            else:
                detail = "alternative_name_match"

        result = self._match_from_record(
            organization,
            status=status,
            status_detail=detail,
            matched_name=matched_name,
            matched_name_types=matched_types,
            matched_candidate=candidate.text,
        )

        self.cache[cache_key] = self._match_to_cache(result)
        return result

    def _validate_original_ror(
        self,
        original_ror: str,
        affiliation_value: str,
        *,
        row: pd.Series,
        bypass_cache: bool,
    ) -> RORMatch:
        normalized_ror = self._normalize_ror_id(original_ror)

        if not normalized_ror:
            return RORMatch(
                status="review",
                status_detail="original_ror_invalid",
            )

        cache_key = self._cache_key(
            "record",
            normalized_ror,
        )

        if not bypass_cache:
            cached = self.cache.get(cache_key)

            if cached is not None:
                record = self._match_from_cache(cached)

                return self._validate_original_record(
                    record,
                    affiliation_value,
                    row=row,
                )

        short_id = normalized_ror.rsplit("/", 1)[-1]

        payload = self._request(
            endpoint=f"https://api.ror.org/v2/organizations/{short_id}",
            params={},
        )

        if isinstance(payload, RORMatch):
            return payload

        record = self._match_from_record(
            payload,
            status="matched",
            status_detail="original_ror_record",
        )

        self.cache[cache_key] = self._match_to_cache(record)

        return self._validate_original_record(
            record,
            affiliation_value,
            row=row,
        )

    def _validate_original_record(
        self,
        record: RORMatch,
        affiliation_value: str,
        *,
        row: pd.Series,
    ) -> RORMatch:
        matched_candidate = ""
        matched_name = ""
        matched_types: tuple[str, ...] = ()

        for candidate in self._generate_exact_candidates(
            affiliation_value
        ):
            candidate_key = self._exact_normalize(candidate.text)

            for name, name_types in record.record_names:
                if self._exact_normalize(name) == candidate_key:
                    matched_candidate = candidate.text
                    matched_name = name
                    matched_types = name_types
                    break

            if matched_candidate:
                break

        if not matched_candidate:
            return RORMatch(
                status="review",
                status_detail="original_ror_conflict",
            )

        if not self._record_matches_country(record, row):
            return RORMatch(
                status="review",
                status_detail="original_ror_country_conflict",
            )

        if record.ror_status == "inactive":
            status = "review"
            detail = "exact_inactive_original_ror"
        elif record.ror_status == "withdrawn":
            status = "review"
            detail = "exact_withdrawn_original_ror"
        else:
            status = "matched"
            detail = "original_ror_validated"

        return replace(
            record,
            status=status,
            status_detail=detail,
            matched_candidate=matched_candidate,
            matched_name=matched_name,
            matched_name_types=matched_types,
        )

    def _compare_with_original_ror(
        self,
        match: RORMatch,
        original_ror: str,
    ) -> RORMatch:
        original = self._normalize_ror_id(original_ror)
        current = self._normalize_ror_id(match.ror_id)

        if not original:
            return replace(
                match,
                status="review",
                status_detail="original_ror_invalid",
            )

        if original == current:
            return match

        return replace(
            match,
            status="review",
            status_detail="ror_not_unique",
        )

    def _may_structure(
        self,
        match: RORMatch,
    ) -> bool:
        if not match.ror_id:
            return False

        if match.status == "matched":
            return True

        return match.status_detail in {
            "exact_inactive_match",
            "exact_withdrawn_match",
            "exact_inactive_original_ror",
            "exact_withdrawn_original_ror",
        }

    def _prepare_output_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        split_counts = (
            df[self.clean_col]
            .fillna("")
            .astype("string")
            .map(
                lambda value: len(
                    self._split_top_level_commas(
                        self._safe_text(value)
                    )
                )
            )
        )

        max_parts = max(
            int(split_counts.max()) if len(split_counts) else 0,
            1,
        )

        pattern = re.compile(
            rf"^{re.escape(self.clean_output_prefix)}_(\d+)$"
        )

        existing_numbers = [
            int(match.group(1))
            for column in df.columns
            if (match := pattern.match(str(column)))
        ]

        if existing_numbers:
            max_parts = max(
                max_parts,
                max(existing_numbers),
            )

        output_columns = [
            f"{self.clean_output_prefix}_{number}"
            for number in range(1, max_parts + 1)
        ]

        for column in output_columns:
            if column not in df.columns:
                df[column] = pd.Series(
                    "",
                    index=df.index,
                    dtype="string",
                )
            else:
                df[column] = (
                    df[column]
                    .astype("string")
                    .fillna("")
                )

        metadata_defaults = {
            self.output_ror_col: "",
            self.ror_status_col: "",
            self.status_col: "pending",
            self.status_detail_col: "",
            self.successor_id_col: "",
            self.successor_name_col: "",
        }

        for column, default in metadata_defaults.items():
            if column not in df.columns:
                df[column] = pd.Series(
                    default,
                    index=df.index,
                    dtype="string",
                )
            else:
                df[column] = (
                    df[column]
                    .astype("string")
                    .fillna(default)
                )

        df[self.status_col] = (
            df[self.status_col]
            .fillna("")
            .astype("string")
            .str.strip()
            .replace("", "pending")
        )

        invalid = ~df[self.status_col].isin(
            self.VALID_STATUSES
        )
        df.loc[invalid, self.status_col] = "pending"

        for index in df.index[
            df[self.status_col].eq("pending")
        ]:
            self._write_initial_split(
                df,
                index,
                self._safe_text(
                    df.at[index, self.clean_col]
                ),
                output_columns=output_columns,
            )

        return output_columns

    def _write_initial_split(
        self,
        df: pd.DataFrame,
        index: Any,
        value: str,
        *,
        output_columns: Sequence[str] | None = None,
    ) -> None:
        columns = list(
            output_columns
            if output_columns is not None
            else self.output_columns
        )

        if not columns:
            return

        for column in columns:
            df.at[index, column] = ""

        for column, part in zip(
            columns,
            self._split_top_level_commas(value),
        ):
            df.at[index, column] = part

    def _write_structured_match(
        self,
        df: pd.DataFrame,
        index: Any,
        match: RORMatch,
        span: OrganizationSpan,
        remaining: Sequence[str],
    ) -> None:
        for column in self.output_columns:
            df.at[index, column] = ""

        if match.ror_status in {"inactive", "withdrawn"}:
            root_name = (
                match.matched_candidate
                or span.matched_text
                or match.matched_name
                or match.display_name
            )
        else:
            root_name = match.display_name

        df.at[index, self.output_columns[0]] = root_name

        for column, value in zip(
            self.output_columns[1:],
            remaining,
        ):
            df.at[index, column] = value

    def _find_organization_span(
        self,
        clean_input: str,
        match: RORMatch,
        *,
        preferred_candidate: str = "",
    ) -> OrganizationSpan | None:
        parts = self._split_top_level_commas(clean_input)

        if not parts:
            return None

        targets = self._deduplicate_preserve_order(
            [
                preferred_candidate,
                match.matched_candidate,
                match.matched_name,
                match.display_name,
                *(
                    name
                    for name, _ in match.record_names
                ),
            ]
        )

        targets = [
            target
            for target in targets
            if self._safe_text(target)
        ]

        if not targets:
            return None

        best: OrganizationSpan | None = None

        for start in range(len(parts)):
            for end in range(start, len(parts)):
                candidate = ", ".join(
                    parts[start : end + 1]
                ).strip()

                score = self._best_locator_score(
                    candidate,
                    targets,
                )

                if score > 0:
                    best = self._prefer_span(
                        best,
                        OrganizationSpan(
                            start=start,
                            end=end,
                            matched_text=candidate,
                            score=score,
                        ),
                    )

                if start == end:
                    embedded = self._find_embedded_target(
                        candidate,
                        targets,
                    )

                    if embedded is not None:
                        target, prefix, suffix = embedded

                        best = self._prefer_span(
                            best,
                            OrganizationSpan(
                                start=start,
                                end=end,
                                matched_text=target,
                                prefix_remainder=prefix,
                                suffix_remainder=suffix,
                                score=1.0,
                            ),
                        )

        return best

    def _best_locator_score(
        self,
        candidate: str,
        targets: Sequence[str],
    ) -> float:
        candidate_key = self._locator_normalize(candidate)

        if not candidate_key:
            return 0.0

        best = 0.0

        for target in targets:
            target_key = self._locator_normalize(target)

            if not target_key:
                continue

            if candidate_key == target_key:
                return 1.0

            best = max(
                best,
                self._locator_similarity(
                    candidate_key,
                    target_key,
                ),
            )

        return best

    def _locator_similarity(
        self,
        left: str,
        right: str,
    ) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())

        left_specific = left_tokens - self.GENERIC_TOKENS
        right_specific = right_tokens - self.GENERIC_TOKENS

        if (
            left_specific
            and right_specific
            and not (left_specific & right_specific)
        ):
            return 0.0

        union = left_tokens | right_tokens

        token_score = (
            len(left_tokens & right_tokens) / len(union)
            if union
            else 0.0
        )

        sequence_score = SequenceMatcher(
            None,
            left,
            right,
        ).ratio()

        containment = 0.0

        if left in right or right in left:
            containment = (
                min(len(left), len(right))
                / max(len(left), len(right))
            )

        return max(
            containment,
            0.65 * token_score + 0.35 * sequence_score,
        )

    def _find_embedded_target(
        self,
        segment: str,
        targets: Sequence[str],
    ) -> tuple[str, str, str] | None:
        for target in sorted(
            targets,
            key=len,
            reverse=True,
        ):
            match = re.search(
                rf"(?<!\w){re.escape(target)}(?!\w)",
                segment,
                flags=re.IGNORECASE,
            )

            if match is None:
                continue

            prefix = self._clean_remainder(
                segment[: match.start()]
            )
            suffix = self._clean_remainder(
                segment[match.end() :]
            )

            return target, prefix, suffix

        return None

    @staticmethod
    def _prefer_span(
        current: OrganizationSpan | None,
        candidate: OrganizationSpan,
    ) -> OrganizationSpan:
        if current is None:
            return candidate

        current_length = current.end - current.start + 1
        candidate_length = candidate.end - candidate.start + 1

        if candidate.score > current.score:
            return candidate

        if (
            candidate.score == current.score
            and candidate_length < current_length
        ):
            return candidate

        return current

    def _remaining_after_span(
        self,
        parts: Sequence[str],
        span: OrganizationSpan,
    ) -> list[str]:
        remaining = list(parts[: span.start])

        if span.prefix_remainder:
            remaining.append(span.prefix_remainder)

        if span.suffix_remainder:
            remaining.append(span.suffix_remainder)

        remaining.extend(parts[span.end + 1 :])

        return self._deduplicate_preserve_order(
            self._clean_remainder(value)
            for value in remaining
            if self._clean_remainder(value)
        )

    def _remove_equivalent_record_names(
        self,
        remaining: Sequence[str],
        match: RORMatch,
    ) -> list[str]:
        name_keys = {
            self._exact_normalize(name)
            for name, _ in match.record_names
            if self._exact_normalize(name)
        }

        if match.display_name:
            name_keys.add(
                self._exact_normalize(match.display_name)
            )

        return [
            value
            for value in remaining
            if self._exact_normalize(value) not in name_keys
        ]

    def _suspect_multiple_organizations(
        self,
        remaining: Sequence[str],
    ) -> bool:
        return any(
            self.STRONG_ORG_RE.search(value)
            and not self.SUBUNIT_START_RE.search(value)
            for value in remaining
        )

    def _generate_exact_candidates(
        self,
        value: object,
    ) -> list[ExactCandidate]:
        text = self._safe_text(value)

        if not text:
            return []

        parts = self._split_top_level_commas(text)

        candidates: list[ExactCandidate] = []

        for start in range(len(parts)):
            for end in range(start, len(parts)):
                span_parts = parts[start : end + 1]
                candidate_text = ", ".join(span_parts).strip()

                if not self._candidate_is_usable(candidate_text):
                    continue

                starts_with_subunit = bool(
                    self.SUBUNIT_START_RE.search(
                        candidate_text
                    )
                )
                strong_org = bool(
                    self.STRONG_ORG_RE.search(
                        candidate_text
                    )
                )
                is_acronym = self._looks_like_acronym(
                    candidate_text
                )

                score = 0

                if strong_org:
                    score += 100
                if starts_with_subunit:
                    score -= 45
                if is_acronym:
                    score += 15

                if len(span_parts) == 1:
                    score += 10
                elif len(span_parts) == 2:
                    score += 20
                else:
                    score += max(
                        2,
                        16 - 4 * (len(span_parts) - 2),
                    )

                score += min(end, 5)

                candidates.append(
                    ExactCandidate(
                        text=candidate_text,
                        score=score,
                        starts_with_subunit=starts_with_subunit,
                        is_acronym=is_acronym,
                    )
                )

        for part in parts:
            for match in re.finditer(
                r"(?<!\w)[A-Z][A-Z0-9.&-]{2,9}(?!\w)",
                part,
            ):
                acronym = match.group(0)

                if self._looks_like_acronym(acronym):
                    candidates.append(
                        ExactCandidate(
                            text=acronym,
                            score=35,
                            starts_with_subunit=False,
                            is_acronym=True,
                        )
                    )

        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                len(candidate.text),
            ),
            reverse=True,
        )

        output: list[ExactCandidate] = []
        seen: set[str] = set()

        for candidate in candidates:
            key = self._exact_normalize(candidate.text)

            if key and key not in seen:
                seen.add(key)
                output.append(candidate)

        return output

    def _candidate_is_usable(
        self,
        value: str,
    ) -> bool:
        text = value.strip()

        if len(text) < 2:
            return False

        if self.NON_ORG_START_RE.search(text):
            return False

        compact = re.sub(r"\s", "", text)

        if not compact:
            return False

        digit_ratio = (
            sum(character.isdigit() for character in compact)
            / len(compact)
        )

        return digit_ratio < 0.35

    def _find_exact_record_name(
        self,
        candidate: str,
        organization: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]] | None:
        candidate_key = self._exact_normalize(candidate)

        matches: list[tuple[str, tuple[str, ...]]] = []

        for entry in organization.get("names", []) or []:
            value = self._safe_text(entry.get("value"))

            if (
                value
                and self._exact_normalize(value) == candidate_key
            ):
                types = tuple(
                    self._safe_text(name_type)
                    for name_type in entry.get("types", []) or []
                    if self._safe_text(name_type)
                )
                matches.append((value, types))

        if not matches:
            return None

        matches.sort(
            key=lambda item: (
                "ror_display" in item[1],
                "label" in item[1],
                "alias" in item[1],
                "acronym" not in item[1],
            ),
            reverse=True,
        )

        return matches[0]

    def _filter_records_by_country(
        self,
        records: Sequence[
            tuple[dict[str, Any], str, tuple[str, ...]]
        ],
        row: pd.Series,
    ) -> tuple[
        list[tuple[dict[str, Any], str, tuple[str, ...]]],
        bool,
        bool,
    ]:
        input_code = self._input_country_code(row)

        if input_code:
            matching = [
                record
                for record in records
                if input_code
                in self._record_country_codes(record[0])
            ]
            known = [
                record
                for record in records
                if self._record_country_codes(record[0])
            ]

            if matching:
                return matching, True, False

            if known and len(known) == len(records):
                return [], True, True

            return list(records), False, False

        input_name = self._input_country_name(row)

        if input_name:
            input_key = self._country_normalize(input_name)
            matching = [
                record
                for record in records
                if input_key
                in {
                    self._country_normalize(name)
                    for name in self._record_country_names(
                        record[0]
                    )
                }
            ]
            known = [
                record
                for record in records
                if self._record_country_names(record[0])
            ]

            if matching:
                return matching, True, False

            if known and len(known) == len(records):
                return [], True, True

        return list(records), False, False

    def _record_matches_country(
        self,
        match: RORMatch,
        row: pd.Series,
    ) -> bool:
        input_code = self._input_country_code(row)

        if input_code:
            return input_code in {
                code.upper()
                for code in match.country_codes
            }

        input_name = self._input_country_name(row)

        if input_name:
            input_key = self._country_normalize(input_name)

            return input_key in {
                self._country_normalize(name)
                for name in match.country_names
            }

        return True

    def _match_from_record(
        self,
        organization: Mapping[str, Any],
        *,
        status: str,
        status_detail: str,
        matched_name: str = "",
        matched_name_types: Sequence[str] = (),
        matched_candidate: str = "",
        matching_type: str = "",
    ) -> RORMatch:
        display_name = self._ror_display_name(organization)
        ror_id = self._safe_text(organization.get("id"))
        ror_status = self._safe_text(
            organization.get("status")
        ).casefold()

        successor_id = ""
        successor_name = ""

        for relationship in organization.get(
            "relationships",
            [],
        ) or []:
            if (
                self._safe_text(
                    relationship.get("type")
                ).casefold()
                == "successor"
            ):
                successor_id = self._safe_text(
                    relationship.get("id")
                )
                successor_name = self._safe_text(
                    relationship.get("label")
                )
                break

        record_names = tuple(
            (
                self._safe_text(name.get("value")),
                tuple(
                    self._safe_text(name_type)
                    for name_type in name.get("types", []) or []
                    if self._safe_text(name_type)
                ),
            )
            for name in organization.get("names", []) or []
            if self._safe_text(name.get("value"))
        )

        if not matched_name:
            matched_name = display_name
            matched_name_types = ("ror_display",)

        return RORMatch(
            status=status,
            status_detail=status_detail,
            ror_id=ror_id,
            display_name=display_name,
            ror_status=ror_status,
            matched_name=matched_name,
            matched_name_types=tuple(matched_name_types),
            matched_candidate=matched_candidate,
            matching_type=matching_type,
            successor_id=successor_id,
            successor_name=successor_name,
            record_names=record_names,
            country_codes=tuple(
                sorted(
                    self._record_country_codes(
                        organization
                    )
                )
            ),
            country_names=tuple(
                sorted(
                    self._record_country_names(
                        organization
                    )
                )
            ),
        )

    def _request(
        self,
        *,
        params: Mapping[str, Any],
        endpoint: str = "https://api.ror.org/v2/organizations",
    ) -> dict[str, Any] | RORMatch:
        for attempt in range(1, self.max_retries + 1):
            self._respect_request_interval()

            try:
                response = self.session.get(
                    endpoint,
                    params=dict(params),
                    timeout=self.timeout,
                )
                self._last_request_at = time.monotonic()

                if response.status_code == 404:
                    return RORMatch(
                        status="review",
                        status_detail="original_ror_not_found",
                    )

                if response.status_code == 429:
                    retry_after = self._retry_after_seconds(
                        response
                    )

                    if attempt < self.max_retries:
                        time.sleep(retry_after)
                        continue

                    return RORMatch(
                        status="rate_limited",
                        status_detail="http_429",
                    )

                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, dict):
                    return RORMatch(
                        status="api_error",
                        status_detail="invalid_json_response",
                    )

                return payload

            except requests.Timeout:
                if attempt == self.max_retries:
                    return RORMatch(
                        status="api_error",
                        status_detail="timeout",
                    )

                time.sleep(2 ** (attempt - 1))

            except requests.RequestException as error:
                if attempt == self.max_retries:
                    return RORMatch(
                        status="api_error",
                        status_detail=type(error).__name__,
                    )

                time.sleep(2 ** (attempt - 1))

            except ValueError:
                return RORMatch(
                    status="api_error",
                    status_detail="invalid_json_response",
                )

        return RORMatch(
            status="api_error",
            status_detail="unknown_api_error",
        )

    def _validate_input_columns(
        self,
        df: pd.DataFrame,
    ) -> None:
        required = {
            self.api_input_col,
            self.clean_col,
        }

        if self.original_ror_col is not None:
            required.add(self.original_ror_col)

        missing = sorted(
            column
            for column in required
            if column not in df.columns
        )

        if missing:
            raise KeyError(
                "Required column(s) missing: "
                + ", ".join(missing)
            )

    def _select_rows(
        self,
        df: pd.DataFrame,
        *,
        statuses: Sequence[str] | None,
        max_rows: int | None,
        force: bool,
    ) -> pd.DataFrame:
        selected = df

        if not force:
            requested = (
                set(statuses)
                if statuses is not None
                else set(self.DEFAULT_RETRY_STATUSES)
            )

            invalid = requested - self.VALID_STATUSES

            if invalid:
                raise ValueError(
                    "Invalid status values: "
                    + ", ".join(sorted(invalid))
                )

            selected = selected[
                selected[self.status_col].isin(requested)
            ]

        if max_rows is not None:
            selected = selected.head(int(max_rows))

        return selected.copy()

    def _clear_metadata(
        self,
        df: pd.DataFrame,
        index: Any,
    ) -> None:
        for column in (
            self.output_ror_col,
            self.ror_status_col,
            self.status_detail_col,
            self.successor_id_col,
            self.successor_name_col,
        ):
            df.at[index, column] = ""

    def _write_metadata(
        self,
        df: pd.DataFrame,
        index: Any,
        match: RORMatch,
    ) -> None:
        df.at[index, self.output_ror_col] = match.ror_id
        df.at[index, self.ror_status_col] = match.ror_status
        df.at[index, self.status_col] = match.status
        df.at[index, self.status_detail_col] = match.status_detail
        df.at[index, self.successor_id_col] = match.successor_id
        df.at[index, self.successor_name_col] = match.successor_name

    def _load_client_id(self) -> str:
        environment_value = os.getenv(
            "ROR_CLIENT_ID",
            "",
        ).strip()

        if environment_value:
            return environment_value

        if not self.env_file.exists():
            return ""

        try:
            for line in self.env_file.read_text(
                encoding="utf-8"
            ).splitlines():
                stripped = line.strip()

                if (
                    not stripped
                    or stripped.startswith("#")
                    or "=" not in stripped
                ):
                    continue

                key, value = stripped.split("=", 1)

                if key.strip() == "ROR_CLIENT_ID":
                    return value.strip().strip('"').strip("'")
        except OSError:
            return ""

        return ""

    def _load_cache(
        self,
    ) -> dict[str, dict[str, Any]]:
        if not self.cache_file.exists():
            return {}

        try:
            with self.cache_file.open(
                "r",
                encoding="utf-8",
            ) as handle:
                data = json.load(handle)

            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _atomic_save_cache(self) -> None:
        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self.cache_file.with_suffix(
            self.cache_file.suffix + ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.cache,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temporary, self.cache_file)

    @staticmethod
    def _atomic_save_dataframe(
        df: pd.DataFrame,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        df.to_csv(
            temporary,
            sep=";",
            index=False,
            encoding="utf-8",
        )

        os.replace(temporary, path)

    def _cache_key(
        self,
        request_type: str,
        value: str,
    ) -> str:
        return (
            f"{self.MATCHING_VERSION}:"
            f"{request_type}:"
            f"{self._request_normalize(value)}"
        )

    @staticmethod
    def _match_to_cache(
        match: RORMatch,
    ) -> dict[str, Any]:
        return {
            "status": match.status,
            "status_detail": match.status_detail,
            "ror_id": match.ror_id,
            "display_name": match.display_name,
            "ror_status": match.ror_status,
            "matched_name": match.matched_name,
            "matched_name_types": list(
                match.matched_name_types
            ),
            "matched_candidate": match.matched_candidate,
            "matching_type": match.matching_type,
            "successor_id": match.successor_id,
            "successor_name": match.successor_name,
            "record_names": [
                [name, list(name_types)]
                for name, name_types in match.record_names
            ],
            "country_codes": list(match.country_codes),
            "country_names": list(match.country_names),
            "cached_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    @staticmethod
    def _match_from_cache(
        data: Mapping[str, Any],
    ) -> RORMatch:
        return RORMatch(
            status=str(data.get("status", "unmatched")),
            status_detail=str(
                data.get("status_detail", "cached_result")
            ),
            ror_id=str(data.get("ror_id", "")),
            display_name=str(data.get("display_name", "")),
            ror_status=str(data.get("ror_status", "")),
            matched_name=str(data.get("matched_name", "")),
            matched_name_types=tuple(
                data.get("matched_name_types", []) or []
            ),
            matched_candidate=str(
                data.get("matched_candidate", "")
            ),
            matching_type=str(
                data.get("matching_type", "")
            ),
            successor_id=str(
                data.get("successor_id", "")
            ),
            successor_name=str(
                data.get("successor_name", "")
            ),
            record_names=tuple(
                (
                    str(item[0]),
                    tuple(item[1] or []),
                )
                for item in data.get("record_names", []) or []
                if isinstance(item, (list, tuple))
                and len(item) == 2
            ),
            country_codes=tuple(
                data.get("country_codes", []) or []
            ),
            country_names=tuple(
                data.get("country_names", []) or []
            ),
        )

    def _country_cache_context(
        self,
        row: pd.Series,
    ) -> str:
        code = self._input_country_code(row)

        if code:
            return f"code={code}"

        name = self._input_country_name(row)

        if name:
            return (
                "name="
                + self._country_normalize(name)
            )

        return "country=none"

    def _input_country_code(
        self,
        row: pd.Series,
    ) -> str:
        if (
            self.country_code_col is None
            or self.country_code_col not in row.index
        ):
            return ""

        return self._safe_text(
            row.get(self.country_code_col)
        ).upper()

    def _input_country_name(
        self,
        row: pd.Series,
    ) -> str:
        if (
            self.country_name_col is None
            or self.country_name_col not in row.index
        ):
            return ""

        return self._safe_text(
            row.get(self.country_name_col)
        )

    @staticmethod
    def _record_country_codes(
        organization: Mapping[str, Any],
    ) -> set[str]:
        values: set[str] = set()

        for location in organization.get("locations", []) or []:
            details = location.get(
                "geonames_details",
                {},
            ) or {}

            code = str(
                details.get("country_code", "")
            ).strip().upper()

            if code:
                values.add(code)

        return values

    @staticmethod
    def _record_country_names(
        organization: Mapping[str, Any],
    ) -> set[str]:
        values: set[str] = set()

        for location in organization.get("locations", []) or []:
            details = location.get(
                "geonames_details",
                {},
            ) or {}

            name = str(
                details.get("country_name", "")
            ).strip()

            if name:
                values.add(name)

        return values

    @staticmethod
    def _ror_display_name(
        organization: Mapping[str, Any],
    ) -> str:
        for name in organization.get("names", []) or []:
            if "ror_display" in (name.get("types", []) or []):
                return str(
                    name.get("value", "")
                ).strip()

        return ""

    @classmethod
    def _normalize_ror_id(
        cls,
        value: object,
    ) -> str:
        text = cls._safe_text(value)

        if not text:
            return ""

        match = cls.ROR_ID_RE.search(text)

        if match is None:
            bare = cls.ROR_ID_BARE_RE.fullmatch(text)

            if bare is None:
                return ""

            identifier = bare.group(1)
        else:
            identifier = match.group(1)

        return (
            "https://ror.org/"
            + identifier.casefold()
        )

    @staticmethod
    def _exact_normalize(
        value: object,
    ) -> str:
        if value is None or pd.isna(value):
            return ""

        text = unicodedata.normalize(
            "NFKC",
            str(value),
        )
        text = (
            text.replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
            .replace("’", "'")
            .replace("`", "'")
        )
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,;:])", r"\1", text)
        text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)

        return text.strip(" \t\r\n.,;:")

    @staticmethod
    def _locator_normalize(
        value: object,
    ) -> str:
        if value is None or pd.isna(value):
            return ""

        text = unicodedata.normalize(
            "NFKD",
            str(value),
        )
        text = "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )
        text = text.casefold().replace("&", " and ")
        text = re.sub(
            r"[^\w]+",
            " ",
            text,
            flags=re.UNICODE,
        )

        return " ".join(text.split())

    @staticmethod
    def _country_normalize(
        value: object,
    ) -> str:
        if value is None or pd.isna(value):
            return ""

        text = unicodedata.normalize(
            "NFKD",
            str(value),
        )
        text = "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )
        text = text.casefold()
        text = re.sub(
            r"[^\w]+",
            " ",
            text,
            flags=re.UNICODE,
        )

        return " ".join(text.split())

    @staticmethod
    def _request_normalize(
        value: object,
    ) -> str:
        if value is None or pd.isna(value):
            return ""

        text = unicodedata.normalize(
            "NFKC",
            str(value),
        )

        return " ".join(text.strip().split())

    @staticmethod
    def _quoted_query(
        value: str,
    ) -> str:
        reserved = re.compile(
            r'([+\-=!(){}\[\]^"~*?:\\/]|&&|\|\|)'
        )
        escaped = reserved.sub(r"\\\1", value)

        return f'"{escaped}"'

    @staticmethod
    def _split_top_level_commas(
        value: str,
    ) -> list[str]:
        if not value:
            return []

        parts: list[str] = []
        current: list[str] = []
        depth = 0

        for character in value:
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

        final = "".join(current).strip()

        if final:
            parts.append(final)

        return parts

    @staticmethod
    def _looks_like_acronym(
        value: str,
    ) -> bool:
        return bool(
            re.fullmatch(
                r"[A-Z][A-Z0-9.\-&]{1,9}",
                value.strip(),
            )
        )

    @staticmethod
    def _clean_remainder(
        value: str,
    ) -> str:
        text = re.sub(
            r"^\s*[\-–—:;/|]+\s*",
            "",
            value,
        )
        text = re.sub(
            r"\s*[\-–—:;/|]+\s*$",
            "",
            text,
        )
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip(" ,;|-–—")

    @staticmethod
    def _deduplicate_preserve_order(
        values: Iterable[str],
    ) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()

        for value in values:
            clean = str(value).strip()
            key = clean.casefold()

            if clean and key not in seen:
                seen.add(key)
                output.append(clean)

        return output

    def _respect_request_interval(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_request_interval - elapsed

        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _retry_after_seconds(
        response: requests.Response,
    ) -> float:
        value = response.headers.get(
            "Retry-After",
            "",
        )

        try:
            return max(float(value), 1.0)
        except (TypeError, ValueError):
            return 30.0

    @staticmethod
    def _safe_text(
        value: object,
    ) -> str:
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        return str(value).strip()
