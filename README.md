# Retrieval-augmented generation

Here's some demo code to customize LLMs with documents from your private knowledge base.

This code is an adaptation of the method and code from this [blog](https://medium.com/@manthapavankumar11/customize-large-language-models-using-langchain-part-1-4731427532a1). The general idea is that you throw your private docs into a vector database, in this case Chroma. At query time, you fetch a few of the most relevant documents and feed them to the LLM to wordsmith into a relevant answer.
