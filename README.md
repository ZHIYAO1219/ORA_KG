# ORA_KG -- Textual Knowledge Graph Construction based on Operations Research
## This is a project exploring possibility of OR in NLP problem. We will target our goal as "Shrink the course Knowledge Graph(KG) and summarize the keyword of courses through Multi-Objective Optimization.<br>

## Here's a preview of our project <br>
There are a bunch of class info on NTU course website. A case and a point,
![image](https://github.com/ZHIYAO1219/ORA_KG/assets/45808654/6c722101-94b5-40e2-9797-45d94c58f1be)
![image](https://github.com/ZHIYAO1219/ORA_KG/assets/45808654/ca40a7c6-77f3-4caf-b8d2-34a87121a697)
We can crawl the courses' name and the corresponding text easily through our **crawler module** .
<br>

Next, leveraging our **preprocessing module**, we can tokenize the the introduction text of the cources. In our example, there will be a list, which may seems like ['student', 'learn', 'methodology', **'operations research'**[1], application,...]. We consider these words as keywords, or having a relationship --"represent"-> to the course on course KG.<br>
[1]How we tokenze words like 'operation research' other than single words will be explain in **n-gram module**<br>

As we can see, there are lots of redundent words, such as 'student', 'learn', if we are dicussing **"keywords"** of a course. We introduce our **Multi-Objective Oprimization model** to explore another approach to replace traditional **Keyword Extraction** methods.<br>
Those module and our model will be introduced in the following article. 

## Background and Motivation
To the best of our knowledge, there are few published papers discuss about this kind of topic-Utilizing operations research techniques on NLP problems. Therefore, we are interested about how these direction would evolve. There are tons of task may be defined as NLP problems, we chose "**Keyword Extraction** on course KG" as our topic because <br>
1. It provides user of the system an intuitive and innovative approach to search other than retrieving the preferavble result only when type in the specific same words in class name.
2. It provides a autometic way to show the keywords for the classes, further make the browsing process more efficient.<br>

## Problem Definition
This study proposes a Multi-Objective Oprimization model to maximize the information value for the course KG and minimize the redundent words to make the words on it truly keywords at the same time.





