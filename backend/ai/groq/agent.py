from ai.analysis.context import load_business_context
from ai.analysis.selector import select_context
from ai.analysis.facts import calculate_facts
from ai.analysis.prompt import build_analysis_prompt
from ai.groq.client import ask_groq


def run_agent():

    print("Loading business context...")

    context = load_business_context()

    print("Selecting relevant context...")

    selected_context = select_context(
        context,
        category="Furniture",
        region="Europe"
    )

    print("Calculating verified facts...")

    facts = calculate_facts(selected_context)

    print("Building grounded prompt...")

    prompt = build_analysis_prompt(
        facts=facts,
        context=selected_context
    )

    print("Prompt size:", len(prompt), "characters")

    print("Sending analysis to Groq...")

    result = ask_groq(prompt)

    return result


if __name__ == "__main__":

    print("=" * 60)
    print("BUSINESS AI AGENT")
    print("=" * 60)

    result = run_agent()

    print("\nAI ANALYSIS")
    print("=" * 60)

    print(result)