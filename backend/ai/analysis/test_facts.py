from ai.analysis.context import load_business_context
from ai.analysis.selector import select_context
from ai.analysis.facts import calculate_facts


def main():

    print("=" * 60)
    print("FACTS TEST")
    print("=" * 60)

    # Load all available data
    full_context = load_business_context()

    # Select only Furniture + Europe
    selected_context = select_context(
        full_context,
        category="Furniture",
        region="Europe"
    )

    # Calculate verified facts
    facts = calculate_facts(selected_context)

    print("\nVERIFIED FACTS")
    print("=" * 60)

    for section, values in facts.items():

        print(f"\n{section}")
        print("-" * 40)

        for key, value in values.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()