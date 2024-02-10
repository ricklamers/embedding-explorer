import wikipediaapi
import requests
import re

def download_and_save_random_pages(num_pages, filename):
    """Downloads a specified number of random Wikipedia pages and saves the text in sentences."""
    wiki_api = wikipediaapi.Wikipedia(
        language='en',
        user_agent="MyWikipediaDownloaderScript/1.0 (Contact: ricklamers@gmail.com)" 
    )

    with open(filename, 'w') as file:
        for _ in range(num_pages):
            # 1. Get a random Wikipedia page title
            random_page_url = 'https://en.wikipedia.org/wiki/Special:Random'
            response = requests.get(random_page_url)
            response.raise_for_status()  # Raise an error if the request fails
            redirect_url = response.url  # Extract the redirected URL with the actual page title
            page_title = redirect_url.split('/')[-1]

            # 2. Fetch content using wikipedia-api
            try:
                page = wiki_api.page(page_title)

                sentences = re.split(r'(?<=[.!?]) +', page.text)
                for sentence in sentences:
                    if sentence:  # Check if sentence is not empty
                        file.write(sentence.strip() + ".\n" if sentence[-1] not in ".!?" else sentence.strip() + "\n")

            except wikipediaapi.WikipediaException as e:
                print(f"Error fetching page '{page.title}': {e}")


# Example usage
num_pages_to_download = 50
filename = 'sentences.txt'
download_and_save_random_pages(num_pages_to_download, filename)
