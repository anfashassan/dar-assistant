from pathlib import Path

from scraper.darglobal import (infer_country, is_project_path, is_valid_project, page_type,
                               parse_project, parse_specs)
from scraper.extract import normalize_url, parse_html
from scraper.http import is_challenge
from scraper.wasalt import (canon_city, is_crawlable, is_property_url, parse_listing, spread_sample,
                            wp_post_to_page)

FX = Path(__file__).parent / "fixtures"


def test_darglobal_project_page():
    page = parse_html((FX / "darglobal_project.html").read_text(), "https://darglobal.co.uk/trump-tower-jeddah")
    assert "cookies" not in page["text"].lower()           # boilerplate removed
    assert "Projects" not in page["text"].split("\n")[0]    # nav removed
    p = parse_project(page)
    assert p["name"] == "Trump Tower Jeddah"
    assert p["property_type"] == "Seafront Residences"
    assert p["completion"] == "December 2029"
    assert p["floors"] == "47"
    assert p["area_sqm"] == "70 to 365"
    assert p["country"] == "Saudi Arabia"
    assert len(p["faqs"]) == 2
    assert "https://darglobal.co.uk/neptune" in page["links"]
    assert not any(l.endswith(".pdf") for l in page["links"])


def test_spec_layouts():
    specs = parse_specs("Property Type\nBranded Villas\n- Status: Under Development\nArea (SQM)300 - 448\nAreas of interest blah")
    assert specs["property_type"] == "Branded Villas"
    assert specs["status"] == "Under Development"
    assert specs["area"] == "300 - 448"


def test_wasalt_listing_page():
    url = "https://wasalt.sa/en/property/sale/villa-360-sqm-facing-west-on-15m-width-street-231985"
    l = parse_listing(parse_html((FX / "wasalt_listing.html").read_text(), url))
    assert l["id"] == "231985" and l["purpose"] == "sale" and l["property_type"] == "Villa"
    assert l["city"] == "Riyadh" and l["district"] == "Al Yasmeen"
    assert l["price_sar"] == 4_800_000 and l["bedrooms"] == 7 and l["bathrooms"] == 8
    assert l["land_area_sqm"] == 360 and l["building_year"] == 2024
    assert l["amenities"] == ["Maids Room", "Garden", "Annex"]
    assert l["ref_no"] == "944378"


LIVE_RENT_TEXT = """For Rent

Apartment 119.17 SQM with 3 Bedrooms

Al Hammraa , Khobar Map view

- 3

 bedrooms
- 119.17 SQM

 built-up area

38,000

/Year

Additional Information

Property usage Residential

Facade 3 Streets

## Property Description

Apartment for Rent in Al Hammraa, Khobar

Built up Area: 119.17 SQM

The property has 3 bedrooms

Building year: 2026

Price: 38000 SAR

Ref no. 723754 | Last updated: 143 days ago

Rent now pay later

3,451.67 */Month*
"""


def test_wasalt_live_layout_rent_listing():
    # layout observed on wasalt.sa in Nov 2026: period on its own line below the price
    url = "https://wasalt.sa/en/property/rent/apartment-11917-sqm-with-3-bedrooms-5879383"
    l = parse_listing({"url": url, "text": LIVE_RENT_TEXT, "title": "Apartment 119.17 SQM with 3 Bedrooms For Rent"})
    assert l["purpose"] == "rent" and l["property_type"] == "Apartment"
    assert l["city"] == "Al Khobar" and l["district"] == "Al Hammraa"   # normalised name
    assert l["price_sar"] == 38_000 and l["rent_period"] == "yearly"   # not the pay-later "/Month"
    assert l["built_area_sqm"] == 119.17 and l["bedrooms"] == 3
    assert l["facade"] == "3 Streets" and l["usage"] == "Residential"


def test_rent_url_wins_over_site_header():
    # every live page starts with "Properties for Sale in Saudi Arabia"; a rental used to
    # be stored as a sale, so a yearly rent of SAR 23,000 looked like a sale price
    url = "https://wasalt.sa/en/property/rent/apartment-70-sqm-with-2-bedrooms-5943629"
    text = "Properties for Sale in Saudi Arabia - wasalt.com\n\nFor Rent\n\n23,000\n\n/Year\n\nPrice: 23000 SAR"
    l = parse_listing({"url": url, "text": text, "title": "Apartment 70 SQM with 2 Bedrooms"})
    assert l["purpose"] == "rent" and l["rent_period"] == "yearly"
    sale = parse_listing({"url": url.replace("/rent/", "/sale/"), "text": "Apartment for Rent in X, Riyadh\nPrice: 9 SAR"})
    assert sale["purpose"] == "sale"   # URL beats the description template too


def test_spread_sample_covers_whole_sitemap():
    urls = [f"u{i}" for i in range(1000)]
    s = spread_sample(urls, 10)
    assert len(s) == 10 and s[0] == "u0" and s[-1] == "u900"
    assert spread_sample(urls[:5], 10) == urls[:5]


def test_wasalt_url_rules():
    assert is_property_url("https://wasalt.sa/en/property/sale/villa-200-sqm-176462")
    assert not is_property_url("https://wasalt.sa/en/sale")
    assert is_crawlable("https://wasalt.sa/en/sale")
    assert not is_crawlable("https://wasalt.sa/ar/property/x-1234")
    assert not is_crawlable("https://wasalt.sa/en/login")


def test_wordpress_post_conversion():
    page = wp_post_to_page({"link": "https://blog.wasalt.sa/en/x/", "title": {"rendered": "Riyadh &amp; Jeddah"},
                            "content": {"rendered": "<h2>Intro</h2><p>Hello <b>world</b></p>"},
                            "excerpt": {"rendered": "<p>Short</p>"}, "date": "2026-01-01"})
    assert page["title"] == "Riyadh & Jeddah" and "## Intro" in page["text"] and page["description"] == "Short"


def test_bot_challenge_pages_detected():
    # real interstitials served by darglobal.co.uk (Imperva) and wasalt.sa (Cloudflare)
    assert is_challenge((FX / "incapsula_challenge.html").read_text())
    assert is_challenge((FX / "cloudflare_challenge.html").read_text())
    assert not is_challenge((FX / "darglobal_project.html").read_text())
    assert not is_challenge((FX / "wasalt_listing.html").read_text())
    assert not is_challenge("") and not is_challenge(None)


def test_empty_project_records_rejected():
    # a challenge page parsed as a "project" used to yield a record with no name or specs
    url = "https://darglobal.co.uk/trump-tower-jeddah"
    blank = parse_project(parse_html((FX / "incapsula_challenge.html").read_text(), url))
    assert not is_valid_project(blank)
    real = parse_project(parse_html((FX / "darglobal_project.html").read_text(), url))
    assert is_valid_project(real)
    assert not is_valid_project({"name": "Only a name"})


def test_live_project_layout_fixes():
    # value wrapped onto a second line (live Trump Tower Jeddah page)
    specs = parse_specs("- Unit Type 1, 2, 3, 4 Bedroom Apartments &\n 3, 4 Bedroom Penthouses\n- Location Jeddah")
    assert specs["unit_type"] == "1, 2, 3, 4 Bedroom Apartments & 3, 4 Bedroom Penthouses"
    assert specs["location"] == "Jeddah"
    # tagline H1 -> fall back to <title> (live Trump Mansions page)
    p = parse_project({"url": "https://darglobal.co.uk/rayana/trump-mansions", "title": "Trump Mansions",
                       "h1": "The place to be", "text": "# The place to be\n\nLocation\n\nRayana, Wadi Safar"})
    assert p["name"] == "Trump Mansions"
    # a normal H1 is kept even if the <title> is longer
    p = parse_project({"url": "https://darglobal.co.uk/neptune", "h1": "Neptune, Interiors by Mouawad",
                       "title": "Neptune, Interiors by Mouawad | Branded Villas Riyadh", "text": ""})
    assert p["name"] == "Neptune, Interiors by Mouawad"


def test_faq_heading_is_not_a_location_and_location_decides_country():
    text = "# WELCOME TO AIDA\n\n## Frequently Asked Questions - AIDA\n\nAIDA is 20 minutes from Muscat. Buyers fly in from Dubai."
    p = parse_project({"url": "https://darglobal.co.uk/landing-page/aida", "h1": "WELCOME TO AIDA", "text": text})
    assert "frequently" not in p["location"].lower()
    assert p["country"] == "Oman"   # from the name "AIDA", not the passing "Dubai" mention
    p = parse_project({"url": "https://darglobal.co.uk/x", "h1": "X",
                       "text": "# X\n\nMuscat | Oman\n\nInvestors from Dubai and Riyadh love it."})
    assert p["location"] == "Muscat | Oman" and p["country"] == "Oman"
    # live Les Vagues page: a bare "Location" heading right before the FAQ heading
    p = parse_project({"url": "https://darglobal.co.uk/les-vagues", "h1": "Les Vagues by Elie Saab",
                       "text": "# Les Vagues by Elie Saab\n\nDoha | Qatar\n\n- Property Type Branded Residential Complex"
                               "\n\n## Location\n\n## Frequently Asked Questions\n\nWhat is it?"})
    assert p["location"] == "Doha | Qatar"


def test_project_path_rules():
    for ok in ("/neptune", "/rayana/trump-mansions", "/one-of-one/project-pagani", "/lp/trump-international-qatar"):
        assert is_project_path(ok), ok
    for bad in ("/", "/faq", "/get-in-touch", "/insights", "/pay-online", "/one-of-one",
                "/press/x", "/ar/neptune", "/terms-conditions", "/terms-of-uses", "/privacy-policy"):
        assert not is_project_path(bad), bad


def test_wasalt_city_names_normalised():
    assert canon_city("Aldammam") == "Dammam" and canon_city("Khobar") == "Al Khobar"
    assert canon_city("Riyadh") == "Riyadh" and canon_city("") == ""
    assert canon_city("Makkah Al Mukarramah") == "Makkah" and canon_city("Alttayif") == "Taif"
    assert canon_city("Bariduh") == "Buraydah" and canon_city("Tbwk") == "Tabuk"


def test_helpers():
    assert normalize_url("https://Darglobal.co.uk/projects/#top") == "https://darglobal.co.uk/projects"
    assert infer_country("AIDA, Muscat") == "Oman"
    assert infer_country("Benahavís, Spain") == "Spain"
    assert page_type("https://darglobal.co.uk/press/x", set()) == "news"
    assert page_type("https://darglobal.co.uk/neptune", {"/neptune"}) == "project"
