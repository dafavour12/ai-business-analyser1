from ai.analysis.context import load_business_context
from ai.analysis.selector import select_context
from ai.analysis.facts import calculate_facts


print("=" * 60)
print("ANALYSIS PIPELINE TEST")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD CONTEXT
# ------------------------------------------------------------

print("\n1. Loading business context...")

context = load_business_context()

print("Context loaded successfully.")


# ------------------------------------------------------------
# 2. SELECT RELEVANT CONTEXT
# ------------------------------------------------------------

print("\n2. Selecting relevant context...")

selected_context = select_context(
    context,
    max_monthly_records=12
)

print("Context selected successfully.")


# ------------------------------------------------------------
# 3. CALCULATE VERIFIED FACTS
# ------------------------------------------------------------

print("\n3. Calculating verified facts...")

facts = calculate_facts(selected_context)

print("Facts calculated successfully.")


# ------------------------------------------------------------
# 4. DISPLAY FACTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VERIFIED FACTS")
print("=" * 60)

for key, value in facts.items():

    print(f"\n{key.upper()}")

    print(value)


# ------------------------------------------------------------
# 5. COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS PIPELINE TEST COMPLETE")
print("=" * 60)