"""
Demo: System and User Prompts

This script demonstrates how a 'system' prompt (instructions that define the
assistant's role and behavior) and a 'user' prompt (the user's question) are
combined to form the input to an LLM. It uses a mock assistant that respects
system constraints such as tone, verbosity, and restricted actions.

No external APIs required. The demo shows two scenarios with different
system prompts to illustrate how the assistant's behavior changes.
"""

from typing import Dict


def build_message_sequence(system: str, user: str) -> Dict[str, str]:
	return {"system": system, "user": user}


def mock_assistant(messages: Dict[str, str]) -> str:
	"""Simple mock assistant that uses system instructions to shape the reply.
	Supported system directives (informal mini-language):
	  - tone:formal|casual
	  - verbosity:short|detailed
	  - forbid:{word}
	"""
	system = messages.get("system", "")
	user = messages.get("user", "")

	# parse a few directives
	tone = "formal" if "tone:formal" in system else "casual"
	verbosity = "short" if "verbosity:short" in system else "detailed"
	forbidden = []
	if "forbid:" in system:
		# naive extraction: find 'forbid:word' tokens
		parts = system.split()
		for p in parts:
			if p.startswith("forbid:"):
				w = p.split(":", 1)[1]
				if w:
					forbidden.append(w.lower())

	# produce a base answer depending on the user's question
	base = ""
	if "asthma" in user.lower():
		base = "Common triggers include pollen, dust mites, cold air, exercise, and respiratory infections."
	elif "diabetes" in user.lower():
		base = "Symptoms include frequent urination, excessive thirst, unexplained weight loss, and fatigue."
	else:
		base = "I don't know based on the provided context."

	# apply forbidden filter
	for f in forbidden:
		base = base.replace(f, "[forbidden]")

	# apply tone
	if tone == "formal":
		base = "According to the provided information, " + base
	else:
		base = "In short: " + base

	# apply verbosity
	if verbosity == "short":
		# return just the first clause
		short = base.split(".")[0]
		if not short.endswith('.'):
			short = short + '.'
		return short

	return base


def demo():
	user_q = "What are the common triggers of asthma?"

	# Scenario A: formal, detailed
	system_a = "You are a helpful medical assistant. tone:formal verbosity:detailed"
	msgs_a = build_message_sequence(system_a, user_q)
	print("--- Scenario A (formal, detailed) ---")
	print("System:", system_a)
	print("User:", user_q)
	print("Assistant:", mock_assistant(msgs_a))
	print()

	# Scenario B: casual, short, forbid the word 'pollen'
	system_b = "You are a friendly assistant. tone:casual verbosity:short forbid:pollen"
	msgs_b = build_message_sequence(system_b, user_q)
	print("--- Scenario B (casual, short, forbid 'pollen') ---")
	print("System:", system_b)
	print("User:", user_q)
	print("Assistant:", mock_assistant(msgs_b))


if __name__ == '__main__':
	demo()

