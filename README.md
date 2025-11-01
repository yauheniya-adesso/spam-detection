# SMS Spam Detection with Neural Networks

## Project Overview

This project demonstrates how machine learning can automatically classify text messages as either "ham" (legitimate) or "spam" (unwanted). It's a practical example of **supervised learning** using a neural network.

## What is Supervised Learning?

Supervised learning is like teaching a student with a textbook that has all the answers included. The algorithm learns from **labeled examples** where we already know the correct answer.

**In this project:**
- **Input:** Text messages
- **Labels:** Each message is marked as "ham" or "spam"
- **Goal:** Train a model to predict labels for new, unseen messages

The process works in three stages:
1. **Training:** The model learns patterns from 80% of our data (training set)
2. **Validation:** We test the model on the remaining 20% (test set) to see how well it learned
3. **Prediction:** The trained model can now classify new messages it has never seen

## How Text Processing Works

Text messages can't be fed directly into a neural network—they must first be converted into numbers. This project uses a technique called **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### Step-by-Step Text Processing:

#### 1. **Tokenization**
The message is broken into individual words:
```
"Win a free iPhone now!" → ["Win", "free", "iPhone", "now"]
```

#### 2. **Stop Word Removal**
Common words like "a", "the", "is" are removed because they don't help distinguish spam from ham.

#### 3. **TF-IDF Conversion**
Each message is converted into a vector of 5,000 numbers. Each number represents how important a specific word is to that message:

- **TF (Term Frequency):** How often a word appears in the message
- **IDF (Inverse Document Frequency):** How rare the word is across all messages

**Example:**
- The word "free" appears frequently in spam but rarely in ham → gets a high score
- The word "meeting" appears in both spam and ham → gets a lower score

The result is a numerical "fingerprint" for each message that captures its meaning.

## How the Neural Network Works

A neural network is a mathematical model inspired by how the brain processes information. It consists of layers of interconnected nodes (neurons) that learn to recognize patterns.

### Network Architecture:

```
Input Layer (5,000 numbers)
        ↓
Hidden Layer (512 neurons)
        ↓
Dropout Layer (prevents overfitting)
        ↓
Output Layer (1 neuron)
        ↓
Prediction: Spam or Ham
```

### What Happens Inside:

1. **Input Layer:** Receives the 5,000 TF-IDF values representing a message

2. **Hidden Layer:** 512 neurons process the input, each looking for different patterns
   - Each neuron applies mathematical weights and an activation function (ReLU)
   - Weights are adjusted during training to improve accuracy

3. **Dropout Layer:** Randomly ignores 50% of neurons during training
   - This prevents the network from memorizing training data (overfitting)
   - Makes the model generalize better to new messages

4. **Output Layer:** One neuron produces a probability between 0 and 1
   - Close to 0 = Ham
   - Close to 1 = Spam

## Training Process

The network learns through an iterative process:

1. **Forward Pass:** Input flows through the network to make a prediction
2. **Calculate Error:** Compare prediction with actual label
3. **Backward Pass:** Adjust weights to reduce error
4. **Repeat:** Do this for 5 epochs (complete passes through all training data)

The model uses **binary crossentropy** loss to measure errors and the **Adam optimizer** to efficiently adjust weights.

## Model Performance

After training, the model is evaluated on the test set (messages it has never seen). The classification report shows:

- **Precision:** Of all messages predicted as spam, how many were actually spam?
- **Recall:** Of all actual spam messages, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall (balanced metric)
- **Accuracy:** Overall percentage of correct predictions

## Key Takeaways 

1. **Quality Data is Essential:** The model is only as good as the labeled examples we provide

2. **Text Becomes Numbers:** Advanced processing (TF-IDF) converts human language into mathematical representations

3. **Neural Networks Learn Patterns:** Through repeated exposure, the network identifies which words and combinations indicate spam

4. **Validation Prevents Overfitting:** Testing on unseen data ensures the model works in real-world scenarios

5. **Continuous Improvement:** More data and tuning can improve performance over time

## Running the Project

**Requirements:**
- Python 3.7+
- TensorFlow, pandas, numpy, scikit-learn, matplotlib

**Dataset:**
- SMS Spam Collection [dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 messages)

**Execution:**
```bash
python spam_detection.py
```

The script will train the model and output performance metrics plus sample predictions.

**Output Example:**

```text
Message: I wish things were different. I wonder when i will be able to show you how much i value you. Pls continue the brisk walks no drugs without askin me please and find things to laugh about. I love you dearly.
True label: ✅ ham, Predicted label: ✅ ham
--------------------------------------------------------------------------------
Message: tddnewsletter@emc1.co.uk (More games from TheDailyDraw) Dear Helen, Dozens of Free Games - with great prizesWith..
True label: ❗spam, Predicted label: ❗spam
--------------------------------------------------------------------------------
Message: Oh, then your phone phoned me but it disconnected
True label: ✅ ham, Predicted label: ✅ ham
--------------------------------------------------------------------------------
Message: Nope i'm not drivin... I neva develop da photos lei...
True label: ✅ ham, Predicted label: ✅ ham
--------------------------------------------------------------------------------
Message: FREE UNLIMITED HARDCORE PORN direct 2 your mobile Txt PORN to 69200 & get FREE access for 24 hrs then chrgd@50p per day txt Stop 2exit. This msg is free
True label: ❗spam, Predicted label: ❗spam
--------------------------------------------------------------------------------
Message: PRIVATE! Your 2003 Account Statement for shows 800 un-redeemed S.I.M. points. Call 08715203685 Identifier Code:4xx26 Expires 13/10/04
True label: ❗spam, Predicted label: ❗spam
--------------------------------------------------------------------------------
Message: Hey Boys. Want hot XXX pics sent direct 2 ur phone? Txt PORN to 69855, 24Hrs free and then just 50p per day. To stop text STOPBCM SF WC1N3XX
True label: ❗spam, Predicted label: ❗spam
--------------------------------------------------------------------------------
Message: What type of stuff do you sing?
True label: ✅ ham, Predicted label: ✅ ham
--------------------------------------------------------------------------------
```