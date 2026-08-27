from .context import load_business_context
from .selector import select_context
from .facts import calculate_facts
from .prompt import build_analysis_prompt


def main():

    print("=" * 60)
    print("PROMPT TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load complete business context
    # ------------------------------------------------------------

    context = load_business_context()

    # ------------------------------------------------------------
    # 2. Select only the relevant context
    # ------------------------------------------------------------

    selected_context = select_context(
        context,
        category="Furniture",
        region="Europe"
    )

    # ------------------------------------------------------------
    # 3. Calculate verified facts
    # ------------------------------------------------------------

    facts = calculate_facts(selected_context)

    # ------------------------------------------------------------
    # 4. Build grounded prompt
    # ------------------------------------------------------------

    prompt = build_analysis_prompt(
        facts=facts,
        context=selected_context
    )

    # ------------------------------------------------------------
    # 5. Display result
    # ------------------------------------------------------------

    print("\nPrompt generated successfully.\n")

    print("Prompt length:")
    print(len(prompt), "characters")

    print("\nPrompt preview:")
    print("=" * 60)
    print(prompt[:5000])


if __name__ == "__main__":
    main()