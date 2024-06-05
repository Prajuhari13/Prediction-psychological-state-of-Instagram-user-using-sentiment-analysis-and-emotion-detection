from flask import Flask, request, render_template_string
from apify_client import ApifyClient
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification
import requests
import os
import webbrowser
import json
from PIL import Image
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the sentiment analysis model
classifier = pipeline('sentiment-analysis')

# Load the facial emotion detection model
feature_extractor = ViTFeatureExtractor.from_pretrained('ycbq999/facial_emotions_image_detection')
emotion_model = ViTForImageClassification.from_pretrained('ycbq999/facial_emotions_image_detection')

# Define a list of emotion labels
emotion_labels = ['SAD', 'DISGUST', 'HAPPY', 'ANGER', 'SURPRISE','NEUTRAL','FEAR']  # replace this with your actual labels

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        profile_url = request.form.get('profileUrl')
        
        # Extracting username from profile URL using BeautifulSoup
        try:
            response = requests.get(profile_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            username = soup.find('meta', attrs={'property': 'og:title'})['content']
        except Exception as e:
            print(e)
            return 'An error occurred while fetching the username.', 500

        access_token = 'ENTER_YOUR_APIFY_API'

        # Initialize the ApifyClient with your API token
        client = ApifyClient(access_token)
 
        # Prepare the Actor input
        run_input = {
            "directUrls": [profile_url],
            "resultsType": "posts",
            "resultsLimit": 200,
            "searchType": "hashtag",
            "searchLimit": 1,
            "addParentData": False,
            "includeFollowerCount": True  # Add parameter to include follower count
        }

        # Initialize a dictionary to store the count of each emotion
        emotion_counts = {label: 0 for label in emotion_labels}

        # Initialize a dictionary to store the count of each sentiment
        sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0}

        try:
            # Run the Actor and wait for it to finish
            run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=run_input)

            # Fetch and print Actor results from the run's dataset (if there are any)
            results = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]

            posts_html_left = ''
            posts_html_right = ''
            for i, item in enumerate(results):
                # Download the image
                response = requests.get(item["displayUrl"])
                filename = f'static/image_{i}.jpg'
                with open(filename, 'wb') as f:
                    f.write(response.content)

                # Perform emotion detection on the image
                img = Image.open(filename)
                inputs = feature_extractor(images=img, return_tensors="pt")
                outputs = emotion_model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()

                # Get the predicted emotion label
                predicted_emotion = emotion_labels[predicted_class]

                # Increment the count of the predicted emotion
                emotion_counts[predicted_emotion] += 1

                comments_html = ''
                for comment in item.get('latestComments', []):
                    # Perform sentiment analysis on the comment
                    result = classifier(comment["text"])
                    sentiment = result[0]['label']
                    score = result[0]['score']

                    # Increment the count of the predicted sentiment
                    sentiment_counts[sentiment] += 1

                    # Add the sentiment and score to the comment HTML
                    color = 'green' if sentiment == 'POSITIVE' else 'red'
                    comments_html += f'<p class="comment" style="color: {color};">{comment["text"]} (Sentiment: {sentiment}, Score: {score:.2f})</p>'

                post_html = f'''
                    <div class="post">
                        <div class="grid-item">
                            <img src="{filename}" alt="Post image">
                            <p class="caption">{item["caption"]}</p>
                            <h3>Predicted Emotion</h3>
                            <p>{predicted_emotion}</p>
                        </div>
                        <div class="grid-item">
                            <h3>Comments</h3>
                            <div class="comments">{comments_html}</div>
                        </div>
                    </div>
                '''

                # Split the posts into two columns
                if i < len(results) / 2:
                    posts_html_left += post_html
                else:
                    posts_html_right += post_html

            # Generate a pie chart of the emotion counts
            plt.figure(figsize=(10, 6))
            plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
            plt.title('POST EMOTION ANALYSIS')
            plt.savefig('static/emotion_distribution.png')

            # Determine the psychological state based on the emotion counts
            max_emotion = max(emotion_counts, key=emotion_counts.get)
            if max_emotion == 'HAPPY':
                psychological_state = 'CHEERFUL'
            elif max_emotion == 'SAD':
                psychological_state = 'DEPRESSED'
            elif max_emotion == 'ANGER':
                psychological_state = 'STRESSED or ANXIOUS'
            elif max_emotion == 'FEAR':
                psychological_state = 'ANXIOUS'
            elif max_emotion == 'DISGUST':
                psychological_state = 'DISTURBED'
            else:
                psychological_state = 'NEUTRAL'

            # Save the scraped data to a JSON file
            with open('data.json', 'w') as f:
                json.dump({
                    "username": username,
                    "posts": len(results),
                    "follower_count": run.get('followerCount', 0),  # Add follower count to the JSON data
                    "psychological_state": psychological_state,  # Add psychological state to the JSON data
                    "posts_html_left": posts_html_left,
                    "posts_html_right": posts_html_right
                }, f)

            # Generate a pie chart of the sentiment counts
            plt.figure(figsize=(10, 6))
            colors = ['green' if sentiment == 'POSITIVE' else 'red' for sentiment in sentiment_counts.keys()]
            plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), colors=colors, autopct='%1.1f%%')
            plt.title('COMMENT SENTIMENT ANALYSIS')
            plt.savefig('static/sentiment_distribution.png')

            # Open the display page in a new browser tab
            webbrowser.open_new_tab('http://localhost:3000/display')

            return 'The profile data has been scraped and saved.', 200
        except Exception as e:
            print(e)
            return 'An error occurred while fetching the profile.', 500

    return '''
        <style>
            body {
                background-image: url("/static/hero-bg.jpg");
                background-size: cover;
            }
            .center-form {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 800px;
                height: 300px;
                background-color: rgba(255, 255, 255, 0);
            }
            .center-form input[type="text"] {
                width: 100%;
                padding: 12px 20px;
                margin: 8px 0;
                box-sizing: border-box;
                border-radius: 10px;
            }
            .center-form input[type="submit"] {
                width: 50%;  /* Reduced the width */
                background-color: #4CAF50;
                color: white;
                padding: 14px 20px;
                margin: 8px 0;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                display: block;  /* Added this line to make the button a block element */
                margin-left: auto;  /* Added this line to center the button */
                margin-right: auto;  /* Added this line to center the button */
            }
            /* Add styles for the post and comments columns */
            .grid-container {
                display: grid;
                grid-template-columns: auto auto;
                gap: 10px;
                padding: 10px;
            }
            .grid-item {
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 10px;
            }
        </style>
        <form method="POST" class="center-form">
            <input type="text" name="profileUrl" placeholder="Enter Instagram profile URL" required>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/display', methods=['GET'])
def display():
    # Load the scraped data from the JSON file
    with open('data.json', 'r') as f:
        data = json.load(f)

    return render_template_string("""
        <head>
            <style>
                .post {
                    width: 80%;
                    margin: auto;
                    padding: 20px;
                    border: 1px solid #ccc;
                    border-radius: 10px;
                }
                .post img {
                    width: 250px;
                    height: 250px;
                }
                                .comment {
                    margin: 5px 0;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                /* Add styles for the grid layout */
                .grid-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;  /* Changed this line to create a 2-grid layout */
                    gap: 10px;
                    padding: 10px;
                }
                .grid-item {
                    padding: 20px;
                    border: 1px solid #ccc;
                    border-radius: 10px;
                }
                .logo {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 200px;  /* Adjust this value to change the size of the logo */
                }
                /* Adjust font sizes */
                .heading {
                    font-size: 24px;
                    margin-bottom: 10px;
                }
                .username {
                    font-size: 20px;
                }
                .count {
                    font-size: 18px;
                    margin-bottom: 5px;
                }
            </style>
        </head>
        <body>
            <img src="/static/logo2.jpg" class="logo" alt="Logo">
            <h1 class="heading">Profile Information</h1>
            <h2 class="heading">Username</h2>
            <h1 class="username">{{ username }}</h1>
            <p class="count">Posts: {{ posts }}</p>
            <p class="count">Followers: {{ follower_count }}</p> <!-- Display follower count -->
            <h2 class="heading">EXPECTED PSYCHOLOGICAL STATE</h2>
            <p class="count">{{ psychological_state }}</p>
            <div class="grid-container">
                <div class="grid-item">{{ posts_html_left|safe }}</div>
                <div class="grid-item">{{ posts_html_right|safe }}</div>
            </div>
            <div class="grid-container">
                <div class="grid-item">
                    <img src="/static/emotion_distribution.png" alt="Emotion Distribution">
                </div>
                <div class="grid-item">
                    <img src="/static/sentiment_distribution.png" alt="Sentiment Distribution">
                </div>
            </div>
        </body>
    """, username=data["username"], posts=data["posts"], follower_count=data.get("follower_count", 0), psychological_state=data["psychological_state"], posts_html_left=data["posts_html_left"], posts_html_right=data["posts_html_right"])

if __name__ == '__main__':
    app.run(port=3000)

