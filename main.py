import os
from fbertanalysis import *
import pandas as pd

run_program = True
tokenizer, model = load_model_and_tokenizer()
df_results = analyze_multiple_docs(tokenizer, model)
print('\nHello! Welcome to the Federal Reserve Meeting Minutes sentiment analysis.\n')
while run_program:
    home_input = int(input("Type 0 to see the sentiment trend of all"
                       " articles for the past 2 years. Or press 1 to see the specific articles"
                       " and see the keyword count and sentiment for specific article"))
    if home_input == 0:
        plot_sentiment_trend(df_results)
    elif home_input == 1:
        for file in os.listdir('fed_minutes'):
            if os.path.isfile(os.path.join('fed_minutes', file)):
                print(f"{file}")  # print all file names of scraped fed minutes articles
                print("These are meeting minutes articles for the past 2 years."
                      " For each file, you can either view the sentiment of the file or the keyword count heatmap.")
                user_file_input = input("Type the filename you are interested in here: ")
                analysis_type_input = input("Either type the word 'heatmap' or 'sentiment' here: ")
                if analysis_type_input == 'heatmap':
                    pass
                elif analysis_type_input == 'sentiment':
                    pass
                else:
                    print("Start over again -- yup run it all again and follow directions correctly.")
                    run_program = False

    else:
        print("Start over again -- yup run it all again and follow directions correctly.")
        run_program = False
