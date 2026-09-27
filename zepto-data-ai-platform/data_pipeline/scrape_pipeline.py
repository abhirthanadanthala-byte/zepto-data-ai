import os
import sqlite3
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# 1. PROJECT SETTINGS
# ---------------------------------------------------------

BASE_URL = "http://books.toscrape.com/"
NUM_PAGES = 5
INR_RATE = 105.50

DATA_DIR = "data_pipeline/data"
OUTPUT_DIR = "data_pipeline/outputs"

CSV_FILE = os.path.join(DATA_DIR, "books_clean.csv")
DB_FILE = os.path.join(DATA_DIR, "zepto_books.db")
SQL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sql_query_outputs.txt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. RATING CONVERSION
# ---------------------------------------------------------

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


# ---------------------------------------------------------
# 3. PARSE AVAILABILITY
# ---------------------------------------------------------

def parse_stock_status(text):
    text = text.lower().strip()

    if text.startswith("in stock"):
        return True

    if text.startswith("out of stock"):
        return False

    return None


# ---------------------------------------------------------
# 4. SCRAPE BOOKS
# ---------------------------------------------------------

def scrape_books():

    books = []

    for page_number in range(1, NUM_PAGES + 1):

        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

        print(f"Scraping page {page_number}: {page_url}")

        response = requests.get(page_url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        products = soup.select("article.product_pod")

        print(f"Books found on page: {len(products)}")

        for product in products:

            # Title
            title_element = product.select_one("h3 a")

            if title_element:
                title = title_element.get("title", "").strip()
                detail_url = urljoin(
                    page_url,
                    title_element.get("href", "")
                )
            else:
                title = ""
                detail_url = ""

            # Price
            price_element = product.select_one(".price_color")
            price_text = (
                price_element.get_text(strip=True)
                if price_element
                else ""
            )

            # Availability
            availability_element = product.select_one(".availability")
            availability_text = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else ""
            )

            # Rating
            rating_element = product.select_one("p.star-rating")

            rating_text = ""

            if rating_element:
                classes = rating_element.get("class", [])

                for class_name in classes:
                    if class_name != "star-rating":
                        rating_text = class_name
                        break

            # -------------------------------------------------
            # Category comes from book detail page
            # -------------------------------------------------

            category = ""

            if detail_url:

                detail_response = requests.get(
                    detail_url,
                    timeout=20
                )

                detail_response.raise_for_status()

                detail_soup = BeautifulSoup(
                    detail_response.text,
                    "html.parser"
                )

                breadcrumbs = detail_soup.select(
                    "ul.breadcrumb li"
                )

                if breadcrumbs:

                    category = breadcrumbs[-1].get_text(
                        strip=True
                    )

            books.append({
                "title": title,
                "price_text": price_text,
                "rating_text": rating_text,
                "availability": availability_text,
                "category": category
            })

    return pd.DataFrame(books)


# ---------------------------------------------------------
# 5. CLEAN DATA
# ---------------------------------------------------------

def clean_data(df):

    print("\nCleaning data...")

    # Convert price
    df["price_gbp"] = (
        df["price_text"]
        .str.replace("£", "", regex=False)
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Convert rating
    df["rating"] = df["rating_text"].map(RATING_MAP)

    # Convert availability
    df["in_stock"] = df["availability"].apply(
        parse_stock_status
    )

    # Numeric failures are median-imputed
    price_median = df["price_gbp"].median()

    if pd.isna(price_median):
        price_median = 0.0

    rating_median = df["rating"].median()

    if pd.isna(rating_median):
        rating_median = 0 

    df["price_gbp"] = df["price_gbp"].fillna(
        price_median
    )

    df["rating"] = df["rating"].fillna(
        rating_median
    )

    # Text/boolean parsing failures are dropped
    df = df.dropna(
        subset=["title", "category", "in_stock"]
    )

    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # Fixed conversion rate from project specification
    df["price_inr"] = (
        df["price_gbp"] * INR_RATE
    ).round(2)

    # Keep only required columns
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------
# 6. CREATE SQLITE DATABASE
# ---------------------------------------------------------

def create_database(df):

    print("\nCreating SQLite database...")

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    connection = sqlite3.connect(DB_FILE)

    cursor = connection.cursor()

    cursor.execute("""
        PRAGMA foreign_keys = ON
    """)

    # Categories table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Insert categories
    categories = sorted(df["category"].unique())

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories(category_name)
            VALUES (?)
            """,
            (category,)
        )

    # Category lookup
    category_map = {}

    rows = cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    ).fetchall()

    for category_id, category_name in rows:
        category_map[category_name] = category_id

    # Insert books
    for _, row in df.iterrows():

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]) if pd.notna(row["price_gbp"]) else 0.0,
                float(row["price_inr"]) if pd.notna(row["price_inr"]) else 0.0,
                int(row["rating"]) if pd.notna(row["rating"]) else 0,
                int(row["in_stock"]),
                category_map[row["category"]]
            )
        )

    connection.commit()

    return connection


# ---------------------------------------------------------
# 7. RUN SQL QUERIES
# ---------------------------------------------------------

def run_sql_queries(connection):

    queries = {

        "Query 1 - SELECT WHERE":
        """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4;
        """,

        "Query 2 - ORDER BY":
        """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC;
        """,

        "Query 3 - LIMIT":
        """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """,

        "Query 4 - DISTINCT":
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
        """,

        "Query 5 - BETWEEN":
        """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating BETWEEN 4 AND 5;
        """,

        "Query 6 - JOIN":
        """
        SELECT
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp DESC
        LIMIT 10;
        """
    }

    output_lines = []

    for query_name, query in queries.items():

        print("\n" + "=" * 60)
        print(query_name)
        print("=" * 60)

        result = pd.read_sql(
            query,
            connection
        )

        print(result.to_string(index=False))

        output_lines.append("=" * 60)
        output_lines.append(query_name)
        output_lines.append("=" * 60)
        output_lines.append(query.strip())
        output_lines.append("\nOUTPUT:")
        output_lines.append(result.to_string(index=False))
        output_lines.append("\n")

    # Save SQL strings and outputs
    with open(
        SQL_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(output_lines)
        )

    return queries


# ---------------------------------------------------------
# 8. READ DATA WITH PANDAS AND REPRODUCE JOIN
# ---------------------------------------------------------

def compare_sql_join_with_pandas(connection, queries):

    print("\nComparing SQL JOIN with pandas merge...")

    books_df = pd.read_sql(
        "SELECT * FROM books",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        connection
    )

    # SQL result
    sql_join = pd.read_sql(
        queries["Query 6 - JOIN"],
        connection
    )

    # Pandas merge
    pandas_join = books_df.merge(
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_join = pandas_join[
        [
            "title",
            "category_name",
            "rating",
            "price_gbp"
        ]
    ]

    pandas_join = pandas_join.sort_values(
        ["rating", "price_gbp"],
        ascending=[False, False]
    ).head(10)

    pandas_join = pandas_join.reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)

    print("\nSQL JOIN:")
    print(sql_join.to_string(index=False))

    print("\nPANDAS MERGE:")
    print(pandas_join.to_string(index=False))

    # Compare values
    same_result = sql_join.equals(pandas_join)

    print("\nDo SQL JOIN and pandas merge match?")
    print(same_result)

    comparison_file = os.path.join(
        OUTPUT_DIR,
        "join_comparison.csv"
    )

    combined = pd.concat(
        [
            sql_join.add_prefix("sql_"),
            pandas_join.add_prefix("pandas_")
        ],
        axis=1
    )

    combined.to_csv(
        comparison_file,
        index=False
    )


# ---------------------------------------------------------
# 9. MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("ZEPTO DATA & AI PLATFORM - DATA PIPELINE")
    print("=" * 60)

    # Scrape
    raw_df = scrape_books()

    print("\nTotal scraped books:", len(raw_df))

    # Clean
    clean_df = clean_data(raw_df)

    print("Total cleaned books:", len(clean_df))
    print(
        "Number of categories:",
        clean_df["category"].nunique()
    )

    # Acceptance checks
    assert len(clean_df) >= 60, \
        "Dataset must contain at least 60 books."

    assert clean_df["category"].nunique() >= 3, \
        "Dataset must contain at least 3 categories."

    # Save clean CSV
    clean_df.to_csv(
        CSV_FILE,
        index=False
    )

    print("\nClean dataset saved to:")
    print(CSV_FILE)

    # Database
    connection = create_database(
        clean_df
    )

    print("\nDatabase saved to:")
    print(DB_FILE)

    # SQL queries
    queries = run_sql_queries(
        connection
    )

    # JOIN comparison
    compare_sql_join_with_pandas(
        connection,
        queries
    )

    connection.close()

    print("\n" + "=" * 60)
    print("MODULE 1 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()