Inspired by [Perspective API](https://www.perspectiveapi.com/) and its real-time comment moderation tools, this Nifty Assignment is about improving online conversations by {{ site.description | downcase }}.

Online abuse and harassment stops people from engaging in conversation. One area of focus is the study of negative online behaviors, such as **toxic comments**: user-written comments that are rude, disrespectful or otherwise likely to make someone leave a discussion. Platforms struggle to effectively facilitate conversations, leading many communities to [limit](https://meta.stackexchange.com/q/342779) or [completely shut down](https://en.wikipedia.org/wiki/R/The_Donald#Quarantine,_restriction,_ban_and_successor) user comments. In 2018, the [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both part of Alphabet), organized a public competition called the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to build better machine learning systems for detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.[^1]

[^1]: When the Conversation AI team first built toxicity models, they found that the models [incorrectly learned to associate](https://medium.com/the-false-positive/unintended-bias-and-names-of-frequently-targeted-groups-8e0b81f80a23) the names of frequently attacked identities with toxicity. In 2019, the Conversation AI team ran another competition about [Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), focusing on building models that detect toxicity across a range of diverse conversations.

Toxic comment classification is a special case of a more general problem in machine learning known as **text classification**. Discussion forums use text classification to determine whether comments should be flagged as inappropriate. Email software uses text classification to determine whether incoming mail is sent to the inbox or filtered into the spam folder.[^2]

![Spam email classifier](spam-classifier.png)

[^2]: Google Developers. Oct 1, 2018. Text classification. In Machine Learning Guides. <https://developers.google.com/machine-learning/guides/text-classification>

Summary
: Improve online conversations by {{ site.description | downcase }}.

Topics
: Binary search trees, recursion, lists and maps.

Audience
: CS2.

Difficulty
: All NLP concepts are abstracted away. The programming task (as we envisioned the assignment) focused on 4 binary search tree operations that can be solved by applying problem solving templates taught in class.

Strengths
: Text classification explores the **social implications of computing** since students learn how programming can address (but not solve) problems that have been at the center of popular attention such as [toxic comment classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview). A spam email classification dataset is also included.
: Students can run the **bundled web app** and test their decision tree classifier on any text in realtime. Web apps are uniquely authentic and engaging since students can see their code served beyond their computer screen and accessible to anyone online.
: NLP abstractions allow instructors to **customize learning objectives**. We focus on data structures and algorithms, but other courses can emphasize information and treat the decision tree as an abstraction.

Weaknesses
: The NLP abstractions allow students to focus on the data structures by hiding the details of the data representation. While students can interact with the dataset, the data itself is mostly an abstraction so questions about the relative importance of data vs. code and how they impact society are not explicitly explored.

Dependencies
: No external dependencies. All NLP abstractions and the web app are self-contained and supported by regular JDK.

Variants
: Rather than focus on data structures and algorithms, the assignment could instead focus on datasets and data representation. Organize a Kaggle competition to emphasize effective data representation for decision trees. Or reflect critically on the decisions that may be explicitly or implicitly encoded in algorithms and datasets by examining [unintended bias in toxicity classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).
