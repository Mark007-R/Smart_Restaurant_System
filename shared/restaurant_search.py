from typing import Any, Callable, Iterable, List, Optional, TypeVar

ItemType = TypeVar("ItemType")


def normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def unique_by_key(items: Iterable[ItemType], key_getter: Callable[[ItemType], str]) -> List[ItemType]:
    unique = {}
    for item in items:
        key = key_getter(item)
        if key not in unique:
            unique[key] = item
    return list(unique.values())


def search_restaurants_by_name(
    items: Iterable[ItemType],
    query: str,
    name_getter: Callable[[ItemType], str],
    limit: int = 10,
) -> List[ItemType]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    matched = [item for item in items if normalized_query in normalize_text(name_getter(item))]
    return unique_by_key(matched, name_getter)[:limit]


def filter_restaurant_objects(
    restaurants: Iterable[Any],
    city: Optional[str] = None,
    cuisine: Optional[str] = None,
    min_rating: float = 0.0,
    max_price: Optional[float] = None,
    dietary_requirement: Optional[str] = None,
    ambiance: Optional[str] = None,
) -> List[Any]:
    results = list(restaurants)

    if city:
        normalized_city = normalize_text(city)
        results = [r for r in results if normalize_text(getattr(getattr(r, "location", None), "city", "")) == normalized_city]

    if cuisine:
        normalized_cuisine = normalize_text(cuisine)
        results = [
            r for r in results
            if normalized_cuisine in [normalize_text(c) for c in getattr(r, "cuisines", [])]
        ]

    results = [r for r in results if float(getattr(r, "overall_rating", 0.0)) >= min_rating]

    if max_price is not None:
        results = [r for r in results if float(getattr(r, "average_cost_per_person", 0.0)) <= max_price]

    if dietary_requirement:
        requirement = normalize_text(dietary_requirement)
        if "vegetarian" in requirement:
            results = [r for r in results if bool(getattr(r, "vegetarian_options", False))]
        elif "vegan" in requirement:
            results = [r for r in results if bool(getattr(r, "vegan_options", False))]
        elif "gluten" in requirement:
            results = [r for r in results if bool(getattr(r, "gluten_free_options", False))]

    if ambiance:
        normalized_ambiance = normalize_text(ambiance)
        results = [
            r for r in results
            if normalized_ambiance in [normalize_text(a) for a in getattr(r, "ambiance", [])]
        ]

    return results
