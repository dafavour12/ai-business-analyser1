from ai.analysis.selector import select_context
from ai.analysis.facts import calculate_facts
from ai.analysis.prompt import build_analysis_prompt
from ai.groq.client import ask_groq


def run_agent(
    context,
    category=None,
    region=None
):
    """
    Run the business analysis agent.

    The context is supplied by the API pipeline rather than
    loaded from the development dataset.
    """

    print("Selecting relevant context...")

    selected_context = select_context(
        context,
        category=category,
        region=region
    )

    print("Calculating verified facts...")

    facts = calculate_facts(
        selected_context
    )

    print("Building grounded prompt...")

    prompt = build_analysis_prompt(
        facts=facts,
        context=selected_context
    )

    print(
        "Prompt size:",
        len(prompt),
        "characters"
    )

    print("Sending analysis to Groq...")

    result = ask_groq(prompt)

    return {
        "facts": facts,
        "analysis": result
    }


if __name__ == "__main__":

    from ai.analysis.context import load_business_context

    print("=" * 60)
    print("BUSINESS AI AGENT TEST")
    print("=" * 60)

    print("\nLoading business context...")

    context = load_business_context()

    result = run_agent(context)

    print("\nAI ANALYSIS")
    print("=" * 60)

    print(result["analysis"])