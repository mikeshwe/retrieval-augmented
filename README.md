# cheaseed

Here's some demo code to customize LLMs with career-coaching materials for the cheaseed.com app.

More broadly, you can use this approach to customize an off-the-shelf LLM with documents from your private knowledge base.

This code is an adaptation of the method and code from this [blog](https://medium.com/@manthapavankumar11/customize-large-language-models-using-langchain-part-1-4731427532a1). The general idea is that you throw your private docs into a vector database, in this case Chroma. At query time, you fetch a few of the most relevant documents and feed them to the LLM to wordsmith into a relevant answer.

Here's an example of the Q&A capabilities. The sample query is "How should you ask for professional advice at work?"

First, we retrieve the top k-most similar documents. Here we display the top 3:

<img width="500" alt="top_topics" src="https://github.com/mikeshwe/cheaseed/assets/4237498/6a376de9-09ec-45c3-8341-a7e5a6b47498">

Then, we feed these relevant documents to the gpt-3.5-turbo LLM to create an answer based on this custom information:

<img width="500" alt="answer" src="https://github.com/mikeshwe/cheaseed/assets/4237498/8f22087a-60ec-4f4d-949d-9bd2b22a742d">

Notice that the answer is a clever synthesis of the two most highly ranked documents from the vector database.
