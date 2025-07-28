# AI Email Assistant for Fashion Store

This is a Python-based AI application designed to automate customer email responses for a fashion e-commerce business. The app classifies incoming emails, extracts relevant product information, and generates personalized replies using a combination of a vector store (ChromaDB), a product catalog, and an LLM (via OpenAI API).

## 🔍 Purpose

The application streamlines the customer service workflow by handling two types of emails:

- **Order Requests:** The app identifies product orders, verifies availability, suggests alternatives if necessary, and generates an appropriate response.
- **Product Inquiries:** It searches for relevant products based on the email's content and provides informative replies.

## ⚙️ How It Works

1. **Email Parsing:** Incoming emails are loaded from a Google Sheet and classified as either `order` or `inquiry` using keyword-based filtering or a custom classifier.
2. **Vector Search:** Email content is queried against a ChromaDB collection containing product descriptions to retrieve the most relevant items.
3. **Metadata Filtering:** Results are filtered based on product availability, category, and other metadata (e.g., exclude products already ordered).
4. **Response Generation:** The app uses OpenAI's GPT model to generate a human-like response based on the selected products and the email context.
5. **Google Sheets Integration:** Responses are saved to a separate Google Sheet, avoiding duplicates and tracking handled emails.

## 📦 Technologies Used

- Python 3.10+
- Pandas
- ChromaDB (vector store)
- OpenAI GPT-4 API
- Google Sheets API
- Logging and structured error handling

## 🧠 Key Features

- RAG-style (Retrieval-Augmented Generation) architecture
- Product suggestion based on stock and category
- Response caching to avoid duplicate processing
- Modular structure to support additional email types or product sources
