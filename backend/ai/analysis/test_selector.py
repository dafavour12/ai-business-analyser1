from .context import load_business_context
from .selector import select_context


print("=" * 60)
print("SELECTED CONTEXT")
print("=" * 60)


# Load all business data
context = load_business_context()


# Select only relevant data
selected = select_context(
    context,
    category="Furniture",
    region="Europe"
)


# Display selected data
for key, value in selected.items():

    print(f"\n{key}")
    print("-" * 40)

    if isinstance(value, list):
        print(f"Records: {len(value)}")

        for row in value[:5]:
            print(row)

    else:
        print(value)