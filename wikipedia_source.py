from urllib.parse import quote

def build_url(self, item):
    safe_item = quote(item.replace(" ", "_"), safe="/:")
    article_url = self.BASE_URL + safe_item
    category_url = self.BASE_URL + "Category:" + safe_item
    print(f"Trying URL: {article_url}")  # Debug print
    return article_url, category_url