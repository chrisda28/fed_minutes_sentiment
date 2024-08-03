import os
from fbertanalysis import *

run_program = True
tokenizer, model = load_model_and_tokenizer()
df_results = analyze_multiple_docs(tokenizer, model)
print('\nHello! Welcome to the Federal Reserve Meeting Minutes sentiment analysis.\n')

while run_program:
    home_input = int(input("Type 0 to see the sentiment trend of all articles for the past 2 years.\n"
                           " Type 1 to see a keyword count heatmap pertaining to all scraped articles.\n"
                           " Type 2 to see the specific articles and sentiment for specific article.\n"))

    if home_input == 0:
        plot_sentiment_trend(df_results)    # display sentiment trend across articles spanning past 2 years
    elif home_input == 1:
        plot_keyword_heatmap(df_results, KEYWORDS)  # display heatmap of keyword counts for all articles
    elif home_input == 2:
        for file in os.listdir('fed_minutes'):
            print(f"{file}")  # print all file names of scraped fed minutes articles so user can choose specific file
        print("These are meeting minutes articles for the past 2 years."
              " For each file, you can either view the sentiment of the specific article.")
        user_file_input = input("Type the exact filename you are interested in seeing sentiment for here: ")
        keyword_count, sentiment = analyze_doc(f"fed_minutes/{user_file_input}", tokenizer, model, KEYWORDS)
        # finding sentiment of specific article user specified above
        if sentiment == 0:
            print("Negative Sentiment")
        elif sentiment == 1:
            print("Neutral Sentiment")
        elif sentiment == 2:
            print("Positive Sentiment")

        else:
            print("Could not find sentiment try again and make sure there aren't any typos.")

    else:
        print("Start over again and make sure there are no typos.")

