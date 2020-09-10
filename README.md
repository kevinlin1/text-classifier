# [Text Classifier](https://kevinl.info/text-classifier/)

Inspired by [Perspective API](https://www.perspectiveapi.com/) and its real-time comment moderation tools, this Nifty Assignment is about improving online conversations by implementing a decision tree data type for text classification.

Online abuse and harassment stops people from engaging in conversation. One area of focus is the study of negative online behaviors, such as **toxic comments**: user-written comments that are rude, disrespectful or otherwise likely to make someone leave a discussion. Platforms struggle to effectively facilitate conversations, leading many communities to [limit](https://meta.stackexchange.com/q/342779) or [completely shut down](https://en.wikipedia.org/wiki/R/The_Donald#Quarantine,_restriction,_ban_and_successor) user comments. In 2018, the [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both part of Alphabet), organized a public competition called the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to build better machine learning systems for detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.

Toxic comment classification is a special case of a more general problem in machine learning known as **text classification**. Discussion forums use text classification to determine whether comments should be flagged as inappropriate. Email software uses text classification to determine whether incoming mail is sent to the inbox or filtered into the spam folder [1].

![Spam email classifier](docs/spam-classifier.png)

## Getting started

The simplest way to use the text classifier is through the [web app](https://toxicity-classification.herokuapp.com/). The first visit may take 30-60 seconds for the server to wake up and train the model.

The [assignment specification](https://courses.cs.washington.edu/courses/cse143/20au/text-classifier/) ([spec.mhtml](docs/spec.mhtml)) contains all of the information that students need to get started. To run the project locally, download the code and implement (or stub) each required method in the `TextClassifier` class.

Compile and run the `Main` class to compute the classifier's training accuracy.

```java
javac -cp ".:lib/*" Main.java && java -cp ".:lib/*" Main
```

Compile and run the `Server` class to launch the [Nifty Web App](https://kevinl.info/nifty-web-apps/).

```java
javac -cp ".:lib/*" Server.java && java -cp ".:lib/*" Server
```

A JUnit 5 `TextClassifierTest` class is provided, though it requires a `GoodTextClassifier` reference solution with a modified `print` method that returns the expected string result.

## Warning

Machine learning models are trained on human-selected and human-generated datasets. Such models encode and reproduce the implicit bias inherent in the datasets. This model is not meant to generalize beyond the toy training datasets. Don’t use this in a real system! When the Conversation AI team first built toxicity models, they found that the models [incorrectly learned to associate](https://medium.com/the-false-positive/unintended-bias-and-names-of-frequently-targeted-groups-8e0b81f80a23) the names of frequently attacked identities with toxicity. In 2019, the Conversation AI team ran another competition about [Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), focusing on building models that detect toxicity across a range of diverse conversations.

The included vectorization algorithms also encode explicit bias. The vectorization ignores all grammar and syntax, treating each occurrence of a word as independent from all other words in the text. Any usage of a word, no matter the context, is considered equally toxic or spammy.

The provided datasets contain text that may be considered profane, vulgar, or offensive.

## References

1. Google Developers. Oct 1, 2018. Text classification. In Machine Learning Guides. <https://developers.google.com/machine-learning/guides/text-classification>
