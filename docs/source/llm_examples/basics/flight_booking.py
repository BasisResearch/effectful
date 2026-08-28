"""Flight booking: composing agents by passing one's typed output to the next.

Demonstrates:
- A standalone ``@Skill.define`` (``extract_flights``) whose typed result is
  stored as an ``Agent`` field and spliced into a second agent's prompt
- A post-condition on a skill's return type: a ``pydantic.AfterValidator``
  that compares the answer against the arguments the call was made with, so
  rejecting a wrong answer and retrying is the harness's job
- A pre-condition on a skill's parameter, guarding it with a second skill's
  judgement: an off-topic seat request is rejected before any booking happens
- Interactive human-in-the-loop flow
- ``Agent`` history for conversational seat selection
"""

import argparse
import dataclasses
import datetime
import enum
from typing import Annotated

import annotated_types
import pydantic

from effectful.handlers.llm import Skill

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
    price: Annotated[int, pydantic.Field(gt=0)]
    origin: Airport  # three-letter airport code
    destination: Airport  # three-letter airport code
    date: datetime.date  # YYYY-MM-DD


class Seat(enum.StrEnum):
    """Seats A and F are window seats. Seats C and D are aisle seats."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


@pydantic.dataclasses.dataclass(frozen=True)
class SeatPreference:
    """
    User's seat preference extracted from natural language.

    Row 1 is the front row with extra legroom.
    Rows 14 and 20 also have extra legroom.
    """

    row: Annotated[int, pydantic.Field(ge=1, le=30)]
    seat: Seat


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
# Extraction skill (inner "agent")
# ---------------------------------------------------------------------------


@Skill.define
def extract_flights(web_page_text: str) -> list[FlightDetails]:
    """Extract all flight details from the following text.

    {web_page_text}
    """


# ---------------------------------------------------------------------------
# Post-condition on the search result (plain Python, no LLM needed)
# ---------------------------------------------------------------------------


def matches_request(
    flight: FlightDetails, info: pydantic.ValidationInfo
) -> FlightDetails:
    """Check that the flight the model chose matches the criteria it was asked for.

    A skill's arguments are the validation context its answer is decoded under,
    so a post-condition can compare that answer against the request without
    being closed over it: ``info.context`` holds this call's ``origin``,
    ``destination`` and ``date`` (and ``self``), whichever way the model
    answered.  Raising rejects the answer -- the harness feeds the message back
    and the model tries again, up to ``--num-retries`` times, after which the
    call raises rather than returning a flight nobody asked for.
    """
    request = info.context or {}
    errors = [
        f"{field} should be {request[field]}, got {getattr(flight, field)}"
        for field in ("origin", "destination", "date")
        if getattr(flight, field) != request[field]
    ]
    if errors:
        raise ValueError("; ".join(errors))
    return flight


# ---------------------------------------------------------------------------
# Flight search agent
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FlightFinder:
    """Agent that finds flights matching user criteria."""

    available_flights: list[FlightDetails]

    @Skill.define
    def find_flight(
        self, origin: Airport, destination: Airport, date: datetime.date
    ) -> Annotated[FlightDetails, pydantic.AfterValidator(matches_request)]:
        """
        Find the cheapest flight from {origin} to {destination} on {date}.

        List of available flights (from the web page):
        <flights>{self.available_flights}</flights>
        """


@Skill.define
def is_seat_request(user_input: str) -> bool:
    """
    Determine whether the user's message is about where they want to sit on
    the plane -- a seat, a row, or a seating preference: {user_input}
    Do not use any tools.
    """


class SeatSelector:
    """Agent that extracts seat preferences from natural language."""

    @Skill.define
    def select_seat(
        self, user_input: Annotated[str, annotated_types.Predicate(is_seat_request)]
    ) -> SeatPreference:
        """Extract the user's seat preference from their message.

        {user_input}
        """


# ---------------------------------------------------------------------------
# Booking flow
# ---------------------------------------------------------------------------


def book_flight(
    origin: Airport,
    destination: Airport,
    date: datetime.date,
    interactive: bool = False,
) -> None:
    """End-to-end flight booking with search, validation, and seat selection."""
    searcher = FlightFinder(available_flights=extract_flights(FLIGHTS_PAGE))

    # --- Search (checked by `find_flight`'s post-condition, so an answer that
    # doesn't match the request is rejected and retried before it reaches here) ---
    flight = searcher.find_flight(origin, destination, date)

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
        try:
            seat = selector.select_seat(request)
            print(f"  Seat: row {seat.row}, seat {seat.seat}")
        except pydantic.ValidationError:
            # The pre-condition rejected the message, so the model was never
            # asked to read a seat out of it. The predicate reports only that it
            # failed, so the message a person sees is the caller's to write.
            print(f"  Rejected: {request!r} is not a seat preference.")
            return

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
