# Textual Knowledge Graph Construction based on Operations Research
This is a project exploring the potential of applying Operations Research (OR) techniques to address Natural Language Processing (NLP) problems. The research objective is focused on 'shrinking the Knowledge Graph (KG) of courses and summarizing course keywords through Multi-Objective Optimization.'

## A preview of our project <br>
There is a bunch of class info on the NTU course website. A case and a point,
![image](https://github.com/ZHIYAO1219/ORA_KG/assets/45808654/6c722101-94b5-40e2-9797-45d94c58f1be)
![image](https://github.com/ZHIYAO1219/ORA_KG/assets/45808654/ca40a7c6-77f3-4caf-b8d2-34a87121a697)
We can crawl the courses' names and the corresponding text easily through our **crawler module**.
<br>

Next, leveraging our **preprocessing module**, we can tokenize the introduction text of the courses. In our example, there will be a list, which may seem like ['student', 'learn', 'methodology', **'operations research'**[1], application,...]. We consider these words as keywords, or having a relationship --"represent"-> to the course on course KG.<br>
[1]How we tokenize words like 'operation research' other than single words will be explained in **n-gram module**.<br>

As we can see, there are lots of redundent words, such as 'student', and 'learn', if we are discussing **"keywords"** of a course. We introduce our **Multi-Objective Optimization model** to explore another approach to replace traditional **Keyword Extraction** methods.<br>
Those modules and our model will be introduced in the following article. **Multi-Objective Optimization model** will be presented in Chapter 2. The supplementary function, such as **crawler module**, **preprocessing module**, **n-gram module**, will be demonstrated in Chapter 3.

## Table of Contents
1. Introduction
2. Methodology
3. Data collention, prerocessing and Analysis Result
4. Conclusion
5. Reference

## 1. Background and Motivation
### 1.1 Background and Motivation
To the best of our knowledge, there are few published papers discussing this kind of topic utilizing operations research techniques on NLP problems. Therefore, we are interested in how these directions would evolve. There are tons of tasks that may be defined as NLP problems, we chose "**Keyword Extraction** on course KG" as our topic because <br>
1. It provides the user of the system an intuitive and innovative approach to search other than retrieving the preferable result only when typing in the specific same words in the class name.
2. It provides an automatic way to show the keywords for the classes, further making the browsing process more efficient.<br>

### 1.2 Problem Definition
This study proposes a Multi-Objective Optimization model to maximize the information value for the course KG and minimize the redundant words to make the words on it truly keywords at the same time.

## 2. Methodology
### 2.1 Research framework
Our research framework involves several key steps. Initially, we collect documents, including their titles and contents, through web crawling. Subsequently, we preprocess these documents to prepare them for analysis and then feed the relevant information into our Multi-Objective Optimization model. The final output of our framework consists of critical words extracted from the documents, which are then used to construct a comprehensive knowledge graph. In essence, our project leverages OR for information extraction, ultimately contributing to the construction of a knowledge graph.

<img src="framework.png" alt="image" width="600">

Note that the collected documents should include the titles and contents, which, in this study, refer to the course names and their corresponding course summaries. Titles and keywords of contents represent the 'entity' of a knowledge graph.

<img src="example_of_elements.png" alt="image" width="300">

### 2.2 Document pre-processing
The document preprocessing techniques are utilized first to simplify texts and help reduce the modeling complexity, which can further improve the modeling efficiency and reduce noise.

**Translate into English**
-    All the documents are firstly translated into English form.
-    `from googletrans import Translator`
`translator = Translator()`

**Stemming**
-    We convert the words to the same basic form, in order to reduce vocabulary diversity.
-    `import nltk`
`stemmer = stem.PorterStemmer()`

**Remove stopwords**
-    We remove stopwords such as Preposition, Pronouns, or Auxiliary Verbs, etc.
-    `import nltk`
`nltk.download('stopwords')`
`stop_words = set(stopwords.words('english'))`

**Remove punctuation**
-    `import string`
`string.punctuation`

**Compute word vector**
-    All the words are converted into vector space using Word2Vector, which is a Word Embedding method and helps us consider the semantic representation and contextual relationships.
-    `from gensim.models import Word2Vec`
`model = Word2Vec(....)`

### 2.3 Solver:
We opted for the Gurobi solver, acknowledging two limitations that need to be addressed during the model formulation:
-   Gurobi can only solve quadratic or lower-degree programming.
-   The denominator should be a constant, not a variable.

### 2.4 Model formulation
**Sets and indices**
-   $I$: unique words
-   $i$ or $k$: a certain word
-   $\mathbf{i}$ or $\mathbf{k}$: a certain word vector
-   $|I|$: number of unique words
-   $J$: documents
-   $j$: a certain document
-   $|J|$: number of documents

**Parameters**
-   $a_{ij}$: number of times word i exists in document j
-   $b_{ij}$: whether word i exists in document j
-   $s_{ik}$: similarity between word i and k
-   $t_{i}$: whether word i appears in more than one document

**Decision variables**
-   $x_{i}$: whether to choose word i in the knowledge graph
-   Other decision variables that make the model solvable by the Gurobi solver:
    -   $A_{j}$
    -   $B_{i}$
    -   $tf_{ij}$
    -   $idf_{i}$
    -   $X_{ik}$

**Objectives and formulas**
-   $obj^{tfidf}$: maximize total TFIDF which is an indicator of the word importance in the documents, so as to maximize the chosen word importance of the whole knowledge graph.
$$TF_{ij} = \frac{a_{ij} x_{i}}{\sum_{i' \in I}{a_{i'j} x_{i'}}}$$
$$IDF_{i} = \ln \frac{|j|}{1 + \sum_{j' \in J}{b_{ij'} x_{i}}}$$
$$TFIDF_{ij} = \frac{a_{ij} x_{i}}{\sum_{i' \in I}{a_{i'j} x_{i'}}} \ln \frac{|j|}{1 + \sum_{j' \in J}{b_{ij'} x_{i}}}$$
TF stands for term frequency, indicating the frequency with which a given word $i$ appears in document $j$. IDF, or inverse document frequency, represents the frequency with which a word $i$ appears in the entire collection of documents. A word with a high TF is deemed more important in a specific document, while a high IDF suggests that the word may be too common and less significant. Therefore, TF-IDF is calculated by multiplying these two indices to strike a balance, providing a measure of a word's importance across the entire set of documents.

-   $obj^{wordnum}$: minimize the number of chosen words, to minimize the nodes that need to be compared with the query to speed up the information-searching process.
  
-   $obj^{sim}$: minimize the sum of word similarities, to maximize the information value.
    -    Cosine similarity: $$s_{ik} = {similarity}(\mathbf{i}, \mathbf{k}) = \frac{\mathbf{i} \cdot \mathbf{k}}{\|\mathbf{i}\| \cdot \|\mathbf{k}\|}$$
    -    Minkowski distance: $$s_{ik} = D(\mathbf{i}, \mathbf{k}) = \left( \sum_{m} \left| i_m - k_m \right|^p \right)^{\frac{1}{p}}$$
For similarity, here we use the cosine similarity and the Minkowski distance, we’ll presents the different results later. Note that here we choose $p$ to be 2, corresponding to the Euclidean distance.

**Constraints**
-   Choose at least $|J|$ words, meaning that the number of chosen words not less than the number of documents: $\sum_{i \in I} x_{i} \geq |J|$
-   Only keep words that can connect different documents to help build the knowledge graph of courses (here we set this constraint as optional since the connecting samples are very limited):
    $t_{i} \geq x_{i}, \forall i \in I$
-   Constraints that form the objective function:
    -   $obj^{tfidf} = \sum_{i \in I, j \in J} tf_{ij} idf_{i}$
    -   $obj^{wordnum} = - \sum_{i \in I} x_{i}$
    -   $obj^{sim} = - \sum_{i \in I, k \in I, i \neq k} s_{ik} X_{ik}$
-   Other constraints that make the model solvable by the Gurobi solver:
    -   $A_{j}(\sum_{i' \in I}{a_{i'j} x_{i'}}) = 1, \forall j \in J$
    -   $B_{i}(1 + \sum_{j' \in J}{b_{ij'} x_{i}}) = |j|, \forall i \in I$
    -   $tf_{ij} = (A_{j})(a_{ij} x_{i}), \forall i \in I, j \in J$
    -   $idf_{i} = \ln (B_{i}), \forall i \in I$
    -   $X_{ik} = x_{i} x_{k}, \forall i \in I, k \in I, i \neq k$

### 2.5 Multi-objective optimization
Several methods for solving multi-objective problems have been explored in past literature. In this study, we adopt the 'Weighted-Sum Method' to integrate our three objectives and approach the problem as a single-objective problem.

-   Weights of the objectives are $w^{tfidf}$, $w^{wordnum}$, $w^{sim}$, with $w^{tfidf} + w^{wordnum} + w^{sim} =1$
-   In the subsequent examples, the three weights are configured to be 1/3 each.

**Weighted-Sum Method**
    $$w^{tfidf} \frac{obj^{tfidf}}{obj^{tfidf, max}} + w^{wordnum} \frac{obj^{wordnum}}{obj^{wordnum, max}} + w^{sim} \frac{obj^{sim}}{obj^{sim, max}}$$

**Lp-Metric Method**
    $$w^{tfidf} \frac{obj^{tfidf}-obj^{tfidf, max}}{obj^{tfidf, max}} + w^{wordnum} \frac{obj^{wordnum}-obj^{wordnum, max}}{obj^{wordnum, max}} + w^{sim} \frac{obj^{sim}-obj^{sim, max}}{obj^{sim, max}}$$

Note that in this case, we normalize the three objectives, making the Weighted-Sum Method and Lp-Metric Method essentially equivalent.

## 3. Data collention, prerocessing and Analysis Result

### 3.1 Case study: links between NTU IM courses
We conducted a case study on the links between courses at National Taiwan University (NTU). We identified certain existing limitations in the course overviews, including a scarcity of prerequisite relationships between courses. This lack of detailed information poses challenges for students seeking to study across different fields and disciplines.

Our research is aimed at addressing these limitations by constructing knowledge graphs for courses within the National Taiwan University's Institute of Information Management (NTU IM). This initiative holds several benefits, such as facilitating the development of a search engine tailored to course information. Additionally, it aids students in comprehending their learning trajectories and provides clarity on cross-disciplinary learning paths.

To simplify the problem, we have selected three courses, and gathered the names of the courses along with their respective course overviews to serve as our documents. It can be noted that only the first two are associated with prerequisites.
1.    Manufacturing Data Science
2.    Statistical Learning and Deep Learning
3.    Web Application Programming


### 3.2 crawler module 
### 3.3 preprocessing module** 
### 3.4 n-gram module

## 4. Conclusion

## 5. Reference
[Multi-objective Optimisation Using Fuzzy and Weighted Sum Approach for Natural Gas Dehydration with Consideration of Regional Climate](https://link.springer.com/article/10.1007/s41660-022-00247-1)




