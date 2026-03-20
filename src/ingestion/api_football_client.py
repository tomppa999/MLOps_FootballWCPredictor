from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests


class ApiError(Exception):
    """Non-retryable error returned in the API response body."""

    def __init__(self, endpoint: str, params: Dict[str, Any], errors: Any) -> None:
        self.endpoint = endpoint
        self.params = params
        self.errors = errors
        super().__init__(f"API errors for {endpoint} params={params}: {errors}")


DEFAULT_BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_BETWEEN_REQUESTS = 0.25
MAX_RETRIES = 4


class ApiFootballClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        sleep_seconds: float = DEFAULT_SLEEP_BETWEEN_REQUESTS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds

        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": api_key})

        host = os.getenv("API_FOOTBALL_HOST")
        if host:
            self.session.headers["x-rapidapi-host"] = host

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Retryable HTTP error {response.status_code}: {response.text[:300]}",
                        response=response,
                    )

                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, dict):
                    raise ValueError(f"Unexpected response shape for {endpoint}: not a dict.")

                api_errors = payload.get("errors")
                if api_errors and api_errors != []:
                    raise ApiError(endpoint, params, api_errors)

                time.sleep(self.sleep_seconds)
                return payload

            except ApiError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                backoff = 2 ** (attempt - 1)
                time.sleep(backoff)

        raise RuntimeError(f"API request failed for endpoint={endpoint}, params={params}") from last_error
    
    def get_league(self, league_id: int) -> Dict[str, Any]:
        return self._get("/leagues", {"id": league_id})

    def get_fixtures_by_league_and_season(
        self,
        league_id: int,
        season: int,
    ) -> Dict[str, Any]:
        return self._get(
            "/fixtures", 
            {
                "league": league_id, 
                "season": season, 
            },
        )

    def get_fixture_statistics(self, fixture_id: int) -> Dict[str, Any]:
        return self._get("/fixtures/statistics", {"fixture": fixture_id})

    def get_team(self, team_id: int) -> Dict[str, Any]:
        return self._get("/teams", {"id": team_id})