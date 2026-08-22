# Commerce Demo

This demo showcases the **offline commerce agent** using synthetic mock data.

**Example query**:
```
I need a laptop under Rs. 60000 for programming
```

### Parsed Requirements
- Category: laptop
- Maximum price: 60000 INR
- Use case: programming
- No specific brand requirement

### Providers Queried
- Amazon Mock
- Flipkart Mock
- Meesho Mock

All three providers are queried in parallel by the `SearchOrchestrator`. Since they are mock providers, the data source for every returned product is `"mock"`.

### Sample Product Results (truncated)
| Provider | Product Title                     | Price (INR) | Rating | Data Source |
|----------|-----------------------------------|------------:|-------:|-------------|
| Amazon   | Dell Inspiron 15 3000 Laptop      | 58,999      | 4.2    | mock |
| Flipkart | Lenovo Ideapad Slim 3 Laptop       | 54,500      | 4.1    | mock |
| Meesho   | HP 15s eq0023AU Laptop            | 59,999      | 4.0    | mock |

### Recommendation
The agent selects the **Lenovo Ideapad Slim 3 Laptop** as the top recommendation because it:
- Meets the price constraint (under 60k INR)
- Has the highest combined score based on price, rating, and feature match for the "programming" use case.
- Provides a good balance of performance and cost.

### Explanation
The recommendation explanation includes:
- **Why this product?** It satisfies all hard constraints (category, price) and scores highest on the weighted scoring formula (price 35%, rating 30%, feature match 20%, etc.).
- **Alternative options** are listed with their scores, showing why they were not selected (higher price, lower rating, or lower feature relevance).
- **Data‑source notice** states that all data is synthetic mock data and not real marketplace information.

> **Note:** This demo runs entirely offline using the pre‑populated mock catalogs. No external API calls, scraping, or live data are involved.
