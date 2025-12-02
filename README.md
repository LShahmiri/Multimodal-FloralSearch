Multimodal Flower Arrangement Query & Image Retrieval

<img width="898" height="444" alt="Multimodal" src="https://github.com/user-attachments/assets/56491529-dfa0-49d1-b04b-82a687ab7696" />

A Vision–Language AI for Flower Search, Matching & Bouquet Recommendations

This project is a multimodal AI application that retrieves the most visually similar flowers to a user's text description and then generates personalized bouquet arrangement suggestions using OpenAI’s GPT-4o vision model.

The system combines:

Text-to-Image Retrieval using ChromaDB + OpenCLIP embeddings

Visual Flower Matching using HuggingFace Flowers-102 dataset

LLM-based Florist Suggestions using GPT-4o (Vision + Text)

Modern Streamlit UI 

AI Query Validator that prevents irrelevant/non-flower queries

1. Text-Based Flower Search
--- 
User enters a natural language description, for example:

“red flowers for my husband birthday”

“yellow flower with round petals”

The system retrieves the most similar flower images.

2. Intelligent Image Retrieval
---
Uses:

OpenCLIP image embedding model

ChromaDB as the vector search engine

Flowers-102 dataset (from HuggingFace)

It loads the dataset, saves images to disk, embeds them, and returns the closest matches.

3. Multimodal LLM (GPT-4o) Recommendations
---
The LLM receives:

User text

Top 2 matched images (as Base64)

And generates:

Elegant bouquet recommendations

Flower combinations

Gift ideas

Personalized messages

4. Query Validator (GPT-4o-mini)
---
Prevents irrelevant requests like:

“hi”

“what is your name”

“play music”

Only flower descriptions are allowed.

