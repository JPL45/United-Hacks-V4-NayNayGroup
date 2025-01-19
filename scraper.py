import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    try:
        # Send HTTP GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find article body using typical HTML tags (can vary per website)
        article_body = soup.find('article') or soup.find('div', {'class': 'article-body'}) or soup.find('div', {'class': 'content'})

        if article_body:
            # Extract and clean up the text from the article body
            paragraphs = article_body.find_all('p')
            article_text = "\n".join([p.get_text() for p in paragraphs])

            return article_text.strip()  # Return cleaned-up text
        else:
            return "Article body not found. Make sure the URL is correct."

    except requests.exceptions.RequestException as e:
        return f"Error while fetching the article: {e}"

if __name__ == "__main__":
    # Take user input for the URL
    url = input("Enter the URL of the article to scrape: ")
    article_text = scrape_article(url)

    # Print or save the article text
    print(article_text)
    
