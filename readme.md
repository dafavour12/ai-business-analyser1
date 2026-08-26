# AI Business Analyser

An AI-powered business analysis platform designed to help businesses understand their sales and operational data, identify trends and problems, and receive actionable business insights.

## Project Goal

The goal of this project is to build a system that can take business data such as sales transactions and use data analysis, machine learning, and AI to answer questions such as:

- What products are performing best?
- Which products are performing poorly?
- How is revenue changing over time?
- What are the major sales trends?
- Which products may require attention?
- What might future sales look like?
- What actions can the business take to improve performance?

## Architecture

```text
Business Data
     |
     v
Data Ingestion
     |
     v
Data Cleaning & Processing
     |
     v
Business Analysis
     |
     +-- Sales Analysis
     +-- Product Analysis
     +-- Customer Analysis
     +-- Inventory Analysis
     +-- Forecasting
     |
     v
AI Analysis / Groq
     |
     v
Business Insights & Recommendations
     |
     v
FastAPI Backend
     |
     v
Next.js Frontend


## Folder Structure

ai-business-analyser/
|
+-- frontend/
|   +-- Next.js application
|
+-- backend/
|   +-- .venv/
|   +-- FastAPI application
|
+-- data/
|   +-- raw/
|   +-- processed/
|
+-- .gitignore
+-- README.md

## Team Development

Frontend Team
    │
    ├── Dashboard
    ├── Charts
    ├── Upload interface
    ├── Reports
    └── AI insights UI

Backend / AI Team
    │
    ├── Data processing
    ├── Business analysis
    ├── Machine learning
    ├── Groq integration
    └── FastAPI