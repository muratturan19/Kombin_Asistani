"""Example usage of the personal outfit assistant."""
from kombin_assistant.database import init_db, add_clothing
from kombin_assistant.assistant import suggest_outfit


def setup_example() -> None:
    init_db()
    add_clothing("images/blue_jeans.jpg", {"type": "pants", "color": "blue", "style": "casual"})
    add_clothing("images/white_shirt.jpg", {"type": "top", "color": "white", "style": "formal"})


if __name__ == "__main__":
    setup_example()
    description = "Akşam şehirde arkadaşlarla buluşma, havada biraz serin"
    print(suggest_outfit(description))
