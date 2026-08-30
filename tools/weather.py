"""
Weather tool — dummy data for now. Equivalent to a @Component bean in Spring.
"""

from langchain_core.tools import tool


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location.

    Use this when the user asks about weather conditions.

    Args:
        location: City name (e.g., "San Francisco", "Amsterdam")

    Returns:
        Weather description string
    """
    return f"The weather in {location} is sunny, 22°C"
