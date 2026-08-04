from abstractions.actor import (
    ActorClass,
    ActorDiedError,
    ActorError,
    ActorRef,
    actor,
    terminate,
)

__all__ = [
    "ActorClass",
    "ActorDiedError",
    "ActorError",
    "ActorRef",
    "actor",
    "terminate",
]


def main() -> None:
    print("Hello from abstractions!")
