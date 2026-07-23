"""Flight booking with multi-agent delegation.

Demonstrates:
- Multi-agent delegation: a tool that internally calls a separate
  ``@Template.define`` (agent-to-agent delegation)
- Programmatic validation of LLM output with retry
- Interactive human-in-the-loop flow
- ``Agent`` history for conversational seat selection
"""

import argparse
import dataclasses
import datetime
import enum
from typing import Literal

from effectful.handlers.llm import Agent, Template

# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------


class Airport(enum.StrEnum):
    SFO = "SFO"
    ANC = "ANC"
    FAI = "FAI"
    JNU = "JNU"
    NYC = "NYC"
    LAX = "LAX"
    ORD = "ORD"
    MIA = "MIA"
    BOS = "BOS"
    SEA = "SEA"
    DFW = "DFW"
    DEN = "DEN"
    ATL = "ATL"
    IAH = "IAH"


@dataclasses.dataclass(frozen=True)
class FlightDetails:
    flight_number: str
    price: int
    origin: Airport  # three-letter airport code
    destination: Airport  # three-letter airport code
    date: datetime.date  # YYYY-MM-DD


@dataclasses.dataclass(frozen=True)
class SeatPreference:
    """
    User's seat preference extracted from natural language.

    Seats A and F are window seats. Seats C and D are aisle seats.
    Row 1 is the front row with extra legroom.
    Rows 14 and 20 also have extra legroom.
    """

    row: int  # 1-30
    seat: Literal["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Sample data (in reality, downloaded from a booking site)
# ---------------------------------------------------------------------------

FLIGHTS_PAGE = """\
1. Flight SFO-AK123 - $350 - San Francisco (SFO) to Anchorage (ANC) - 2025-01-10
2. Flight SFO-AK456 - $370 - San Francisco (SFO) to Fairbanks (FAI) - 2025-01-10
3. Flight SFO-AK789 - $400 - San Francisco (SFO) to Juneau (JNU) - 2025-01-20
4. Flight NYC-LA101 - $250 - New York (NYC) to Los Angeles (LAX) - 2025-01-10
5. Flight ORD-MIA202 - $200 - Chicago (ORD) to Miami (MIA) - 2025-01-12
6. Flight BOS-SEA303 - $120 - Boston (BOS) to Seattle (SEA) - 2025-01-12
7. Flight DFW-DEN404 - $150 - Dallas (DFW) to Denver (DEN) - 2025-01-10
8. Flight ATL-IAH505 - $180 - Atlanta (ATL) to Houston (IAH) - 2025-01-10
"""

# ---------------------------------------------------------------------------
# Extraction template (inner "agent")
# ---------------------------------------------------------------------------


@Template.define
def extract_flights(web_page_text: str) -> list[FlightDetails]:
    """Extract all flight details from the following text.

    {web_page_text}
    """


# ---------------------------------------------------------------------------
# Flight search agent
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FlightFinder(Agent):
    """Agent that finds flights matching user criteria."""

    available_flights: list[FlightDetails]

    @Template.define
    def find_flight(
        self, origin: Airport, destination: Airport, date: datetime.date
    ) -> FlightDetails:
        """
        Find the cheapest flight from {origin} to {destination} on {date}.

        List of available flights (from the web page):
        <flights>{self.available_flights}</flights>
        """


# ---------------------------------------------------------------------------
# Seat selection agent
# ---------------------------------------------------------------------------


class SeatSelector(Agent):
    """Agent that extracts seat preferences from natural language."""

    @Template.define
    def select_seat(self, user_input: str) -> SeatPreference:
        """Extract the user's seat preference from their message.

        {user_input}
        """


# ---------------------------------------------------------------------------
# Validation (plain Python, no LLM needed)
# ---------------------------------------------------------------------------


def validate_flight(
    flight: FlightDetails, origin: Airport, destination: Airport, date: datetime.date
) -> list[str]:
    """Check that the selected flight matches the requested criteria."""
    errors = []
    if flight.origin != origin:
        errors.append(f"origin should be {origin}, got {flight.origin}")
    if flight.destination != destination:
        errors.append(f"destination should be {destination}, got {flight.destination}")
    if flight.date != date:
        errors.append(f"date should be {date}, got {flight.date}")
    return errors


# ---------------------------------------------------------------------------
# Booking flow
# ---------------------------------------------------------------------------


def book_flight(
    origin: Airport,
    destination: Airport,
    date: datetime.date,
    interactive: bool = False,
    max_retries: int = 3,
) -> None:
    """End-to-end flight booking with search, validation, and seat selection."""
    searcher = FlightFinder(available_flights=extract_flights(FLIGHTS_PAGE))

    # --- Search with validation retry ---
    flight = None
    for attempt in range(max_retries):
        candidate = searcher.find_flight(origin, destination, date)
        errors = validate_flight(candidate, origin, destination, date)
        if errors:
            print(f"  [attempt {attempt}] Rejected: {'; '.join(errors)}")
            continue
        flight = candidate
        break

    if flight is None:
        print("Could not find a valid flight.")
        return

    print(
        f"  Found: {flight.flight_number} ${flight.price} "
        f"({flight.origin}->{flight.destination} on {flight.date})"
    )

    # --- User approval (interactive only) ---
    if interactive:
        if input("  Book this flight? (yes/no): ").strip().lower() != "yes":
            print("  Cancelled.")
            return

    # --- Seat selection ---
    selector = SeatSelector()
    seat_requests = (
        [input("  Seat preference: ")]
        if interactive
        else ["I'd like a window seat with extra legroom please"]
    )
    for request in seat_requests:
        seat = selector.select_seat(request)
        print(f"  Seat: row {seat.row}, seat {seat.seat}")

    print(f"  Booked {flight.flight_number}, seat {seat.row}{seat.seat}!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    airports = list(Airport)
    parser.add_argument(
        "--origin",
        type=Airport,
        choices=airports,
        default=Airport.SFO,
        metavar="CODE",
        help="Origin airport code",
    )
    parser.add_argument(
        "--destination",
        type=Airport,
        choices=airports,
        default=Airport.ANC,
        metavar="CODE",
        help="Destination airport code",
    )
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        default=datetime.date(2025, 1, 10),
        metavar="YYYY-MM-DD",
        help="Travel date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with user prompts",
    )
    args = parser.parse_args()

    book_flight(
        origin=args.origin,
        destination=args.destination,
        date=args.date,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
