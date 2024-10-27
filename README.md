# AI-Powered-Web-Text-Scraper-and-Q-A

This project is a powerful tool that combines **LangChain** document loading, **Google Generative AI**, and web scraping with **BeautifulSoup** to fetch and process data from webpages, PDFs, and other formats. It utilizes **FAISS** to store vectorized text data, enabling efficient search and question answering.



https://github.com/user-attachments/assets/c43c97af-88fd-43f1-b2da-0b2c3f3921e6



## Table of Contents

1. [Features](#features)
2. [Setup Instructions](#setup-instructions)
3. [Usage](#usage)
4. [Environment Variables](#environment-variables)
5. [Code Breakdown](#code-breakdown)
6. [License](#license)

## Features

- Web scraping of HTML content using BeautifulSoup.
- Document loading and parsing from various formats (text files, PDFs, etc.).
- Embedding text data using Google Generative AI.
- Vector storage with FAISS for efficient similarity search.
- Question answering using LangChain's QA chain.

## Setup Instructions

To set up the project on your local machine, follow these steps:

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Navigate to the project directory:

    ```bash
    cd project_directory
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up your environment variables by creating a `.env` file. Add your Google API key:

    ```bash
    GOOGLE_API_KEY=your_api_key_here
    ```

5. Run the application:

    ```bash
    python app.py
    ```

## Usage

You can use this project to scrape text data from a website and process it for question answering tasks. Here’s an example:

1. Modify the `link` variable to your desired webpage:

    ```python
    link = "https://example.com"
    ```

2. The script will fetch and process the data, storing it as vectorized embeddings using FAISS.

3. Use the LangChain question answering chain to query the text data.

## Environment Variables

The project requires an environment variable for Google Generative AI:

- `GOOGLE_API_KEY`: Your Google API key to access generative AI services. You can set this in a `.env` file.

## Code Breakdown

### Key Functions

1. **`loader(link)`**:
    - Fetches and parses HTML content from the provided URL using `requests` and `BeautifulSoup`.
    
    ```python
    def loader(link):
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        return text
    ```

2. **`create_text_file(link)`**:
    - Uses the `loader()` function to retrieve text and save it to a file.

3. **Google Generative AI Configuration**:
    - Sets up the API key for Google Generative AI embeddings using environment variables.

    ```python
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    ```

## License

This project is licensed under the MIT License.
