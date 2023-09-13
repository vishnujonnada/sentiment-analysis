from flask import Flask, render_template, request, redirect, url_for, flash
from textblob import TextBlob
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)
app.secret_key = 'tffyfyyyyffyfufu'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    file = request.files['file']
    display_image = False
    image_name = None  # Initialize image_name variable

    # Check if a file is uploaded
    if file:
        try:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            text = ' '.join(df['text'])

            # Check if the DataFrame is empty
            if df.empty:
                flash("The CSV file is empty.", 'error')
                return redirect(url_for('index'))

            # Check if the DataFrame contains a column with text data
            if 'text' not in df.columns:
                flash("The CSV file must contain a column labeled 'text'.", 'error')
                return redirect(url_for('index'))

        except Exception as e:
            flash('Error processing the CSV file: ' + str(e), 'error')

        sentiments = []
        for row in df['text']:
            blob = TextBlob(row)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiments.append('Positive')
            elif polarity < 0:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')

        sentiment_counts = pd.Series(sentiments).value_counts()
        plt.figure(figsize=(8, 6))

        # Define colors for the bars based on sentiment
        colors = ['green' if s == 'Positive' else 'blue' if s == 'Neutral' else 'red' for s in sentiment_counts.index]

        sentiment_counts.plot(kind='bar', color=colors)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)

        static_dir = os.path.join(app.root_path, 'static')
        timestamp = str(int(time.time()))
        image_name = f'sentiment_chart_{timestamp}.png'
        image_path = os.path.join(static_dir, image_name)
        plt.tight_layout()
        plt.savefig(image_path)
        display_image = True

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiment_category = None

    if sentiment > 0:
        sentiment_category = 'Positive'
    elif sentiment < 0:
        sentiment_category = 'Negative'
    else:
        sentiment_category = 'Neutral'

    if sentiment_category == 'Positive':
        sentiment_message = 'Positive'
    elif sentiment_category == 'Negative':
        sentiment_message = 'Negative'
    else:
        sentiment_message = 'Neutral'

    flash(sentiment_message, 'info')

    return render_template('result.html', sentiment_category=sentiment_category, sentiment_message=sentiment_message, display_image=display_image, image_name=image_name,)

if __name__ == '__main__':
    app.run(debug=True)
