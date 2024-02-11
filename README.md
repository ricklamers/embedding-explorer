# PCA Dimensionality Reduction of OpenAI Embeddings

This project demonstrates the use of PCA (Principal Component Analysis) for dimensionality reduction of sentence embeddings generated by OpenAI's API. It visualizes the embeddings in a 3D scatter plot, differentiating between sentences from a common corpus and user-input sentences.

![image](https://github.com/ricklamers/embedding-explorer/assets/1309307/d21da00f-6e11-4130-809a-e9a4112cdc4f)

## Setup

1. **Environment Variables**: Set your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY="your_api_key_here"
   ```
2. **Dependencies**: Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. **Running the App**: Start the Streamlit app with:
   ```
   streamlit run main.py
   ```

## Usage

- Adjust the number of lines to load from the common sentence corpus using the Streamlit slider.
- Enter your own sentences in the provided text area.
- The app will generate embeddings, train the PCA model, and visualize the results in a 3D scatter plot.

## Additional Tools

- **Wikipedia Scraper**: `scrape.py` is included for downloading random Wikipedia pages to use as a common corpus. It saves the sentences to a text file for later use by the main application.

## Procfile

For deployment, a `Procfile` is provided for platforms like Railway/Heroku.
