---
title: Text Classifier
---

# Text Classifier

Inspired by [Perspective API](https://www.perspectiveapi.com/) and its real-time comment moderation tools, this Nifty Assignment is about improving online conversations by {{ site.description | downcase }}.

Online abuse and harassment stops people from engaging in conversation. One area of focus is the study of negative online behaviors, such as **toxic comments**: user-written comments that are rude, disrespectful or otherwise likely to make someone leave a discussion. Platforms struggle to effectively facilitate conversations, leading many communities to [limit](https://meta.stackexchange.com/q/342779) or [completely shut down](https://en.wikipedia.org/wiki/R/The_Donald#Quarantine,_restriction,_ban_and_successor) user comments. In 2018, the [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both part of Alphabet), organized a public competition called the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to build better machine learning systems for detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.

Warning

: Machine learning models are trained on human-selected and human-generated datasets. Such models reproduce the bias inherent in the data. The included data representation algorithms also encode overly-simplistic assumptions about natural language by treating each occurrence of a word as independent from all other words in the text. Any usage of a word, no matter the context, is considered equally toxic. Don't use this in a real system!

  When the Conversation AI team first built toxicity models, they found that the models [incorrectly learned to associate](https://medium.com/the-false-positive/unintended-bias-and-names-of-frequently-targeted-groups-8e0b81f80a23) the names of frequently attacked identities with toxicity. In 2019, the Conversation AI team ran another competition about [Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), focusing on building models that detect toxicity across a range of diverse conversations.

  The provided datasets contain text that may be considered profane, vulgar, or offensive.

Toxic comment classification is a special case of a more general problem in machine learning known as **text classification**. Discussion forums use text classification to determine whether comments should be flagged as inappropriate. Email software uses text classification to determine whether incoming mail is sent to the inbox or filtered into the spam folder.[^1]

![Spam email classifier](spam-classifier.png)

[^1]: Google Developers. Oct 1, 2018. Text classification. In Machine Learning Guides. <https://developers.google.com/machine-learning/guides/text-classification>

Summary
: Improve online conversations by {{ site.description | downcase }}.

Topics
: Data structures, binary trees, binary search trees, recursion, mutation.

Audience
: A second programming course (CS2), a data structures course, or a machine learning course.

Difficulty
: The assignment as presented in CS2 focuses on 4 fundamental binary search tree operations: construction, search, traversal, and removal. These can be solved by applying programming templates taught in class. We provide students all of the NLP and machine learning concepts as abstractions.

Strengths
: Text classification explores a **social implication of computing** since students learn how programming can address (but not fully solve) real-world problems that have been at the center of popular attention. The assignment touches on the limits of computing by pointing out how algorithms are ultimately dependent on the given data.
: Students can run the **bundled web app** and test their decision tree classifier on any text in realtime. Web apps are uniquely authentic and engaging since students can see their code served beyond their computer screen and accessible to anyone online. Students can also host their web app for free online.
: The provided abstractions allow instructors to **customize learning objectives**. We focus on data structures and algorithms, but other courses can flip the abstractions: instructors can provide the decision tree as an abstraction so that students can focus on implementing the machine learning components.

Weaknesses
: The provided abstractions allow students to focus on programming the decision tree data structure by hiding all of the machine learning implementation. While students can interact with the dataset, students likely won't feel that they understand everything about the assignment compared to something they write entirely on their own.

Dependencies
: Beyond the typical treatment of the binary trees in a programming-focused CS2 course, instructors may want to formally introduce decision trees, splitting, pruning, and vectorization beyond the information provided in the sample assignment specification. In terms of software dependencies, the machine learning abstractions depend on 5MB of `jar` files from the [Smile](https://haifengl.github.io/) project.

Variants
: Rather than focus on data structures and algorithms, the assignment could instead focus on datasets and data representation. Instructors can also organize Kaggle competitions to give students exposure to the community and teamwork aspects involved in developing better machine learning algorithms.
: Rather than focus on programming, the assignment could instead focus on computer ethics by investigating [unintended bias in toxicity classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) and holistic approaches to addressing online harm such as [restorative justice](https://www.brookings.edu/techstream/the-promise-of-restorative-justice-in-addressing-online-harm/).
: Rather than assume that the solution to the problem of toxicity involves introducing a machine learning (or even computational) model, the assignment could discuss the broader ecology of the system such as [Squadbox](https://homes.cs.washington.edu/~axz/pub_details.html?id=squadbox): A Tool to Combat Email Harassment Using Friendsourced Moderation. Sometimes, the better solution is one that is not purely computational.

## Getting started

The simplest way to use the text classifier is through the [web app](https://toxicity-classification.herokuapp.com/). The first visit may take 30--60 seconds for the server to wake up and train the model.

The [assignment specification](https://courses.cs.washington.edu/courses/cse143/20au/text-classifier/) ([spec.mhtml](spec.mhtml)) contains all of the information that students need to get started. To run the project locally, [download the code from GitHub]({{ site.github.repository_url }}) and implement (or stub) each required method in the `TextClassifier` class.

Compile and run the `Main` class to compute the classifier's training accuracy.

```java
javac -cp ".:lib/*" Main.java && java -cp ".:lib/*" Main
```

Compile and run the `Server` class to launch the [Nifty Web App](https://kevinl.info/nifty-web-apps/).

```java
javac -cp ".:lib/*" Server.java && java -cp ".:lib/*" Server toxic.tsv
```

A JUnit 5 `TextClassifierTest` class is provided, though it requires a `GoodTextClassifier` reference solution with a modified `print` method that returns the expected string result.

## References
