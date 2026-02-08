"""Human-in-the-loop task planner.

Demonstrates:
- An ``Agent`` that proposes a plan of action steps
- Human approval/rejection of each step before execution
- Feedback from rejection is fed back to the agent via history
- ``@Tool.define`` for executing approved actions
- Non-interactive mode for testing (auto-approves all steps)
"""

import argparse
import dataclasses
import enum
import os

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class ActionType(enum.StrEnum):
    send_email = "send_email"
    create_file = "create_file"
    schedule_meeting = "schedule_meeting"
    done = "done"


@dataclasses.dataclass(frozen=True)
class ProposedAction:
    action: ActionType
    description: str
    details: str


# ---------------------------------------------------------------------------
# Simulated action execution
# ---------------------------------------------------------------------------


execution_log: list[str] = []


@Tool.define
def execute_action(action: ActionType, details: str) -> str:
    """Execute an approved action. Returns a confirmation message."""
    msg = f"[executed] {action}: {details}"
    execution_log.append(msg)
    return msg


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------


class Planner(Agent):
    """Agent that proposes actions one at a time for human approval."""

    @Template.define
    def propose_next(self, task: str, feedback: str) -> ProposedAction:
        """You are a task planner helping the user accomplish a goal.

        Task: {task}

        Feedback from the last step: {feedback}

        Review the conversation history for previously completed actions.
        Propose the next action to take. If the task is complete,
        set action to "done".

        If a previous proposal was rejected, propose something different
        that addresses the feedback.
        """
        raise NotHandled


# ---------------------------------------------------------------------------
# Human-in-the-loop execution
# ---------------------------------------------------------------------------


def run_with_approval(
    task: str, interactive: bool = False, max_steps: int = 5
) -> list[str]:
    """Run a task planner with human approval for each step."""
    planner = Planner()
    feedback = "No actions taken yet. Start planning."

    for step in range(max_steps):
        proposal = planner.propose_next(task, feedback)

        if proposal.action == ActionType.done:
            print(f"  [step {step + 1}] Done: {proposal.description}")
            break

        print(
            f"  [step {step + 1}] Proposed: {proposal.action} - {proposal.description}"
        )
        print(f"           Details: {proposal.details}")

        if interactive:
            answer = input("  Approve? (yes/no + reason): ").strip()
            approved = answer.lower().startswith("y")
        else:
            answer = "yes"
            approved = True

        if approved:
            result = execute_action(proposal.action, proposal.details)
            print(f"  {result}")
            feedback = f"Approved and executed: {result}"
        else:
            print(f"  [rejected] {answer}")
            feedback = f"Rejected: {answer}"

    return list(execution_log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-in-the-loop task planner")
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with human approval prompts",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of action steps",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)

    task = (
        "Organize a team lunch for next Friday. "
        "Send an email to the team, create a shared document for "
        "restaurant suggestions, and schedule a meeting to finalize plans."
    )

    with handler(provider), handler(RetryLLMHandler(num_retries=3)):
        print(f"Task: {task}\n")
        log = run_with_approval(
            task,
            interactive=args.interactive,
            max_steps=args.max_steps,
        )
        print(f"\nExecution log ({len(log)} actions):")
        for entry in log:
            print(f"  {entry}")
