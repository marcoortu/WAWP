import pandas as pd
import json


def sentiment_label(sentiment):
    if sentiment > 0:
        return 'positive'
    if sentiment < 0:
        return 'negative'
    return 'neutral'


def generate_fixtures():
    df = pd.read_csv(
        '../dataset/absita_2018_training.csv',
        delimiter=';'
    )
    column_names = df.columns
    rows = []
    for index, row in df.iterrows():
        sentiment = 0
        json_row = {
            "model": "sentiment_tool.TextPattern",
            "pk": index,
            "fields": {
                "text": row['sentence'],
                "sentiment": ''
            }
        }
        for column_name in column_names:
            if 'positive' in column_name and row[column_name] > 0:
                sentiment += 1
            elif 'negative' in column_name and row[column_name] > 0:
                sentiment -= 1
        json_row['fields']['sentiment'] = sentiment_label(sentiment)
        if index > 5000:
            json_row['fields']['sentiment'] = None
        rows.append(json_row)
    with open('./fixtures/textpattern.json', mode='w') as json_file:
        json_file.write(json.dumps(rows))


if __name__ == '__main__':
    generate_fixtures()
