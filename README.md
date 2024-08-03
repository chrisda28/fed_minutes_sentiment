# Federal Reserve Meeting Minutes Sentiment Analysis
This project scrapes and analyzes Federal Reserve meeting minutes articles for the past two years, including the current
year. It uses a pretrained financial text model (ProsusAI/finBERT) to analyze sentiment and
basic functionality to count specific keywords in the articles.
The results are visualized through various plots and a basic user interface in the Python console.

## Features
- Scrapes Federal Reserve meeting minutes articles for the past two years
- Analyzes sentiment of articles using a pretrained financial text model
- Counts occurrences of specific keywords in the articles

Visualizes results through:
- Keyword count heatmap across all articles
- Sentiment trend plot across all articles
- Allows users to view sentiment for individual articles

## Prerequisites
Python 3.10

## Installation

1. Clone this repository: git clone https://github.com/chrisda28/fed_minutes_sentiment
navigate to project directory

2. Install the required packages:
```pip install -r requirements.txt```


## Usage

1. Run the data scraping script:
```python data.py```
- This will scrape the most current meeting minutes articles and store them in a folder.
2. Then, run the main analysis program:
```python main.py```
- This will start the interactive interface where you can choose to:
- View the keyword count heatmap across all articles
- See the sentiment trend plot across all articles
- Analyze sentiment for a specific article


## Note
- The program may take between 5-20 minutes to run, depending on the amount of data and your system's performance.
- Most meeting minutes articles tend to have a neutral sentiment, likely due to their formal and unbiased wording.
- This project uses the ProsusAI/finbert model for sentiment analysis, which is released under the MIT license.

## Limitations

The sentiment analysis may not capture subtle nuances in the formal language used in Federal Reserve documents.
The keyword counting is case-insensitive and may count partial matches.

## Future Improvements

Implement more sophisticated natural language processing techniques
Expand the range of analyzed documents
Improve the user interface for easier interaction with the data

## AI Usage and Helpful Resources
- used Claude AI to help implement plotting functions, improve readability of code, and ask general questions
- used tutorial code to help implement finBERT model and navigate token limit constraints https://www.youtube.com/watch?v=hgg2GAgDLzA
- used tutorial to get basic understanding of using pretrained models https://www.youtube.com/watch?v=GSt00_-0ncQ
- finBERT model documentation:  https://huggingface.co/ProsusAI/finbert


## Contributing
Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License
This project is for educational and personal use only. While the code itself is not licensed for distribution,
it uses the ProsusAI/finbert model which is released under the MIT license.
