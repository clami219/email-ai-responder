import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
import logging
import chromadb
from openai import OpenAI
import json

logging.basicConfig(level=logging.INFO)

# === Authentication ===
def authenticate_gspread(json_path, scopes):
  creds = Credentials.from_service_account_file(json_path, scopes=scopes)
  return gspread.authorize(creds)

# === Create and initialize worksheet ===
def create_and_init_worksheet(spreedsheet, title, fields):
  try:
    return spreedsheet.worksheet(title)
  except gspread.exceptions.WorksheetNotFound:
    worksheet = spreedsheet.add_worksheet(title=title, rows=50, cols=len(fields))
    worksheet.update([fields], 'A1')
    return worksheet

# === Loading data from Google Sheet ===
def load_data(spreadsheet, worksheet_name):
  return get_as_dataframe(spreadsheet.worksheet(worksheet_name)).dropna(how="all")

# === Classify email ===
def classify_email(openai_client, email_subject, email_message):
   response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an email classification assistant. Classify the email into one of the following categories: 'order', 'inquiry'. Respond with the category only."},
        {"role": "user", "content": f"Identify the category of the following email:\nSubject: {email_subject}\nMessage: {email_message}"}
    ])
   return response.choices[0].message.content

# === Load products in ChromaDB ===
def load_products_to_chromadb(df_products, collection):
    for idx, row in df_products.iterrows():
        product_id = row['product_id']
        
        #Check if product already exists in the collection
        if str(product_id) in collection.get(ids=[str(product_id)])['ids']:
            continue # Skip if product already exists

        # Add product to the vector database
        product_name = row['name']
        product_category= row['category']
        product_description = row['description']
        product_seasons = row['seasons']
        product_price = row['price']
        product_stock = row['stock']

        full_description = f"""{product_name}: {product_description}
        The product (ID: {product_id}) belongs to the category {product_category} and is great in {product_seasons}. The cost of the product is {product_price}.
        """

        # Check if the product already exists in the collection
        existing = collection.get(ids=[product_id])

        # If the product does not exist, add it
        if not existing["ids"]:
            logging.info(f"Adding product {product_name} to the collection.")
            collection.add(
                documents=[full_description],
                metadatas=[{"product_id": product_id, "name": product_name, "category": product_category, "seasons": product_seasons, "price": product_price, "stock": product_stock}],
                ids=[str(product_id)]
            )

# === Generate suborders from email ===  
def generate_suborders(openai_client, query):
    prompt = f"""<INSTRUCTIONS>
Given the following user email, generate a list of independent orders that corresponds to each product the user wants to order through the email. 
For each order, return two fields: product_data, a text containing all the information provided by the customer in regard to the specific product required (product name, product id, product description, etc.); quantity: the desired quantity in numerical format.
Make sure the customer wants to buy the product, based on the information provided in the email.
Return the result as JSON with the key "data" and the list of orders as its value.
If no products are identified, return a JSON with the key "data" and the list of orders as an empty list.
Only output valid JSON and no other characters.
Double check the JSON format and ensure it is valid.
Do not output markdown backticks. Just output raw JSON only.
</INSTRUCTIONS>

<QUERY>{query}</QUERY>
"""
    response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an order processing assistant. Generate suborders based only on the user's email."},
        {"role": "user", "content": prompt}
    ])
    try:
        content = response.choices[0].message.content
        suborder_dict = json.loads(content)
        return suborder_dict['data']
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {e}")
        return []
    
# === Process order request to get product_id and quantity for each order ===
def process_order_request(email_data, suborders, relevant_products, openai_client):
    email_id = email_data['email_id']
    email_subject = email_data['subject']
    email_message = email_data['message']

    prompt = f"""<INSTRUCTIONS>
Given the following user email, generate a list of orders that corresponds to each product the user wants to order through the email. 
To help in identifying the orders there is a hypotetical list of orders that have already been identified from the email and that should be considered in the identification of the orders.
If the hypotetical list of orders is empty, just use the email subject and message to identify the products, using solely the products present in the product list provided.
The identification of the products should be based solely on the products present in the product list provided.
For each order, return two fields: product_id, the id corresponding to the product that the user wants to order, taken from the products list provided; quantity: the user desired quantity for the product in numerical format.
If the hypotetical orders are provided, always use the quantity identified within the hypotetical list of orders provided.
If no hypotetical orders are provided, identify the quantity based on the email subject and message.
If unsure about the quantity and the hypotetical order is not empty and has a valid numerical quantity, use that one as default.
There must be only one order for each product, with the total quantity required for that specific product. The quantity must be a number and should not include any other information.
If it is not possible to identify the quantity as a number, return 1 as the default quantity.
If the user has provided a quantity range, return the maximum value of the range as the quantity if the availability in stock allows it.
If the user has provided a quantity range and the maximum value of the range is not available in stock, return the maximum value of the range that is available in stock and that is greater than the minimum in the range.
If the user has provided a quantity range and the minimum value in the range is uqual to or greater than the available stock, use the minimum value in the range as quantity.
If it is not possible to identify a product, skip it.
If no products are identified, return an empty list.
If you are unable to identify the product id for a specific product, skip the product from the order list.
If the description provided in the email by the user does not clearly match any product in the products list provided, skip that product from the order list.
If the quantity in stock is not sufficient to fulfill the order, the quantity should be kept the same and not diminished to send a partial order.
Return the result as JSON with the key "data" and the list of orders as its value.
Only output valid JSON and no other characters.
Double check the JSON format and ensure it is valid.
Do not output markdown backticks. Just output raw JSON only.
</INSTRUCTIONS>

<EMAIL>
<SUBJECT>{email_subject}</SUBJECT>
<MESSAGE>{email_message}</MESSAGE>
</EMAIL>

<ORDERS HYPOTESIS>
    {json.dumps(suborders)}
</ORDERS HYPOTESIS>
<PRODUCTS LIST>
    {relevant_products}
</PRODUCTS LIST>

"""
    response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an order processing assistant. Generate a list of orders based only on the user's email, on the orders hypotesis, and on the products list provided."},
        {"role": "user", "content": prompt}
    ])
    try:
        content = response.choices[0].message.content
        orders_dict = json.loads(content)
        return orders_dict['data']
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {e}")
        return []

# === Generate order response ===
def generate_order_response(email_data, order_data, relevant_products, openai_client):
    email_subject = email_data['subject']
    email_message = email_data['message']

    prompt = f"""<INSTRUCTIONS>
Generate a professional response to the user based on the order data provided. The response should confirm the order and provide details about the products ordered.
If all the products are out of stock, inform the user that the order cannot be processed.
If some products are out of stock, inform the user about the products that are out of stock and confirm the order for the available products.
If some products are out of stock, you may suggest to the user alternative products among the relevant products provided that are available in stock in the required quantity and that are similar to the products ordered which are missing.
If among the alternative products there is nothing that is similar to the product out of stock (check the product description for this), don't propose alternative products.
If some products are out of stock and there are no alternatives available or appropriate, encourage the user to wait for restock.
The response should be concise and informative, addressing the user's email directly.
The signature at the bottom of the email, after the greetings, should include only the text "Customer Service Team" and no other information or placeholder.
The tone of the email should be professional and friendly.
All prices are intended in US dollars and are shown using the specific symbol ($).
The email should be structured with a clear subject line, a greeting, the body of the email, and a closing signature.
The body of the email should include the order details in a table, with columns for product ID, product name, quantity, and status.
At the end of the order details there should be the total cost for the order, calculated as the sum of the total price for each product, which is given by the cost of the single product multiplied by its quantity.
If the quantity required was not fully available, specify this to the user, indicating the quantity that has been included in the order and the quantity that was not available.
If the order data is empty do not include the order details table, but just a message apologizing for the inconvenience and asking for more information to identify the products.
If the order data is empty, do not include the total cost in the email.
If the order data is empty, do not include the order details table.
If there is a range in the quantity written in the email content, consider it above the quantity present in the hypotetical orders provided.
If the user has provided a quantity range, return the maximum value of the range as quantity if the availability in stock is enough to cover for it.
If the user has provided a quantity range and the maximum value of the range is not available in stock, return the maximum value of the range that is available in stock and that is greater than or equal to the minimum in the range.
If the user has provided a quantity range and the minimum value in the range is uqual to or greater than the available stock, use the minimum value in the range as quantity.
If, and only if, there are some products out of stock, include a short sencente to apologize for the inconvenience and a section for alternative products with their details.
If there are some products out of stock, select accurately the alternative products to suggest, ensuring they are exactly of the same kind as the required product and available in stock, otherwise do not propose any alternative product.
If all the products required are available (no product required out of stock), do not include the sentence of apology nor the section for alternative products.
If no alternative products are available and none of the products are out of stock, do not include apologies in the message.
If the order data is empty or it is not possible to identify any product from the email, return an apologetic response where you explain that it was not possible to identify the specified products and that we need more information to knwo exactly the product they desire, encouraging them to visit our online store (no need to put a placeholder for this).
Only for products present in the order data a order can be created.
No product not present in the order data can be included in the ordered products specified in the email.
If the order data is empty, do not include the order details table.
If the order data is empty, do not include the total cost in the email.
If no product is available in stock, do not include the total cost in the email.
If only one product is ordered and the quantity is 1, do not specify that the total cost was calculated multiplying the quantity by the unit price, just provide the total cost.
If the quantity in stock is not sufficient to fulfill the order, the quantity should be kept the same and not diminished to send a partial order and the order should be marked as out of stock.
If the quantity in stock is enough (even equal to the quantity required), the order should be marked as created and processed accordingly.
If no fitting alternative products are available, do not include the section for alternative products.
If all products are available and in stock, avoid mentiontioning alternative products in the reply.
Only output valid HTML and no other characters.
Double check the HTML format and ensure it is valid.
Do not output markdown backticks. Just output raw HTML only.
</INSTRUCTIONS>

<EMAIL>
<SUBJECT>{email_subject}</SUBJECT>
<MESSAGE>{email_message}</MESSAGE>
</EMAIL>

<ORDER DATA>
{''.join([
    f"""<ORDER>
    <PRODUCT ID>{x.get('product_id')}</PRODUCT ID>
    <QUANTITY>{x.get('quantity')}</QUANTITY>
    <STATUS>{x.get('status')}</STATUS>
    <UNIT PRICE>{x.get('price')}</UNIT PRICE>+
    <CURRENTLY IN STOCK>{x.get('currently_in_stock')}</CURRENTLY IN STOCK>
</ORDER>""" for x in order_data
])}
</ORDER DATA>

<RELEVANT PRODUCTS>
{relevant_products}
</RELEVANT PRODUCTS>
"""
    response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a professional seller assistant, generating professional answers to customers orders via email. You generate a response to the user based on the order data provided."},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content

# === Generate inquiry response ===
def generate_inquiry_response(email_data, relevant_products, openai_client):
    email_subject = email_data['subject']
    email_message = email_data['message']

    prompt = f"""<INSTRUCTIONS>
Generate a professional response to the user based on the email inquiry. The response should answer the customer's questions, based solely on the relevant products data.
If the inquiry is about a specific product, provide detailed information about that product.
If the inquiry is about a general topic, provide relevant information based on the products information available.
The response should be concise and informative, addressing the user's email directly.
The signature at the bottom of the email, after the greetings, should include only the text "Customer Service Team" and no other information or placeholder.
The tone of the email should be professional and friendly.
The email should be structured with a clear subject line, a greeting, the body of the email, and a closing signature.
If the information required is not available, inform the user that we need more information to provide a precise answer or suggest to visit our online store for more details.
Only output valid HTML and no other characters.
Double check the HTML format and ensure it is valid.
Do not output markdown backticks. Just output raw HTML only.
</INSTRUCTIONS>

<EMAIL>
<SUBJECT>{email_subject}</SUBJECT>
<MESSAGE>{email_message}</MESSAGE>
</EMAIL>

<RELEVANT PRODUCTS>
{relevant_products}
</RELEVANT PRODUCTS>
"""
    response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a professional seller assistant, generating professional answers to customers inquiries via email. You generate a response to the user based solely on the product data provided."},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content

def main():
    # Configuration
    ACCESS_KEY_PATH = "access_key.json"
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1jDlayp5eUY2kWNouKAkFvqgygQ55XZ6CKydQTCHkfqI"
    OPENAI_KEY = "#######################"

    openai_client = OpenAI(
       base_url='https://47v4us7kyypinfb5lcligtc3x40ygqbs.lambda-url.us-east-1.on.aws/v1/',
       api_key=OPENAI_KEY
    )

    # Authentication and setup
    client = authenticate_gspread(ACCESS_KEY_PATH, SCOPES)
    spreadsheet = client.open_by_url(SPREADSHEET_URL)
    sheet_products = spreadsheet.worksheet("products")
    sheet_email_classification = create_and_init_worksheet(spreadsheet, "email-classification", ['email ID', 'category'])
    sheet_order_status = create_and_init_worksheet(spreadsheet, "order-status", ['email ID', 'product ID', 'quantity', 'status'])
    sheet_order_response = create_and_init_worksheet(spreadsheet, "order-response", ['email ID', 'response'])
    sheet_inquiry_response = create_and_init_worksheet(spreadsheet, "inquiry-response", ['email ID', 'response'])

    # Loading data
    df_emails = load_data(spreadsheet, "emails")
    df_email_classification = load_data(spreadsheet, "email-classification")

    logging.info("Data loaded successfully.")

    # Checking for existing classifications
    classified_ids = set(df_email_classification['email ID'].astype(str))

    # Classifying emails
    for idx, row in df_emails.iterrows():
        email_id = row['email_id']
        email_subject = row['subject']
        email_message = row['message']
        if email_id in classified_ids: 
            continue # Skip already classified emails
        email_category = classify_email(openai_client, email_subject, email_message)
        # Update email classification
        sheet_email_classification.append_row([email_id, email_category])

    # Loading products data in the vector database
    client_chroma = chromadb.PersistentClient()
    collection = client_chroma.get_or_create_collection(
        name="products_data",
        metadata={"hnsw:space":"cosine"}
    )
    df_products = load_data(spreadsheet, "products")
    load_products_to_chromadb(df_products, collection)

    logging.info("Products loaded into ChromaDB.")

    # Processing order requests
    df_email_merge = df_email_classification.merge(
        df_emails, left_on='email ID', right_on='email_id', how='inner'
    )
    order_requests = df_email_merge[df_email_merge['category'] == 'order']
    
    # Load order response data to identify already processed orders
    df_order_response = load_data(spreadsheet, "order-response")

    order_requests = order_requests.merge(
        df_order_response, left_on='email_id', right_on='email ID', how='left'
    )

    # Filter out orders that have already been processed
    pending_orders = order_requests[order_requests['response'].isnull()]

    # Processing unprocessed orders
    for idx, row in pending_orders.iterrows():
        email_message = row['message']
        email_subject = row['subject']
        email_id = row['email_id']

        logging.info(f"####Processing order for email ID {email_id}...")

        # Generate suborders from email
        suborders = generate_suborders(openai_client, f"Subject: {email_subject}\nMessage: {email_message}")
        if not suborders:
            logging.warning(f"No suborders generated for email ID {email_id}.")
        
        suborders_results = ""
        relevant_products = ""

        logging.info(f"######Suborders generated: {suborders}")
        # Process each suborder to find relevant products
        for suborder in suborders:
            product_data = suborder.get('product_data', {})
            quantity = suborder.get('quantity', 1)

            found_products = collection.query(
                query_texts=[f"{product_data}"],
                n_results=3
            )
            if found_products['documents'][0]:
                suborders_results += f"<PRODUCT ORDER><PRODUCT DESCRIPTION>{product_data}</PRODUCT DESCRIPTION><PRODUCT QUANTITY>{quantity}</PRODUCT QUANTITY></PRODUCT ORDER>"    
                for idx,product in enumerate(found_products['documents'][0]):
                    relevant_products += f"""
<PRODUCT>
    <PRODUCT DATA>{product}</PRODUCT DATA>
    <PRODUCT ID>{found_products['ids'][0][idx]}</PRODUCT ID>
    <AVAILABLE IN STOCK>{found_products['metadatas'][0][idx]['stock']}</AVAILABLE IN STOCK>
</PRODUCT>"""

        # If no suborders were generated, try to find products based on email subject and message
        if not suborders:
            logging.warning(f"No suborder were identified for email ID {email_id}.")
            logging.info("Trying to find products based on email subject and message...")
            found_products = collection.query(
                query_texts=[f"{email_subject} - {email_message}"],
                n_results=10
            )
            for idx,product in enumerate(found_products['documents'][0]):
                relevant_products += f"""
<PRODUCT>
    <PRODUCT DATA>{product}</PRODUCT DATA>
    <PRODUCT ID>{found_products['ids'][0][idx]}</PRODUCT ID>
    <AVAILABLE IN STOCK>{found_products['metadatas'][0][idx]['stock']}</AVAILABLE IN STOCK>
</PRODUCT>"""
                
        # Generating the order data providing the order data acquired
        order_data = process_order_request(
            email_data=row,
            suborders=suborders_results,
            relevant_products=relevant_products,
            openai_client=openai_client
        )
        
        logging.info(f"######Order data generated for email ID {email_id}...")
        logging.info(order_data)
        logging.info("-------")

        order_status = []
        alternative_products = ""

        # Update order status and response
        for order in order_data:
            product_id = order.get('product_id')
            quantity = int(order.get('quantity', 1))

            product_row = df_products[df_products['product_id'] == product_id]
                                
            if not product_row.empty:
                product_stock = int(product_row['stock'].iloc[0])
            else:
                product_stock = 0

            if product_stock is not None and product_stock >= quantity:
                status = 'created'
            else:
                status = 'out of stock'

            # Collect order status
            order_status.append({
                'product_id': product_id,
                'quantity': quantity,
                'status': status,
                'price': product_row['price'].iloc[0] if not product_row.empty else 0,
                'currently_in_stock': product_stock
            })
            # Insert order status
            sheet_order_status.append_row([email_id, product_id, quantity, status])

            if status == 'created':
                # Find the row index of the product in the dataframe
                row_index = df_products.index[df_products['product_id'] == product_id].tolist()
                if row_index:
                    idx = row_index[0]
                    # Updating dataframe
                    df_products.at[idx, 'stock'] -= quantity
    
                    # Update Google Sheet with new stock values
                    sheet_row = idx + 2  # +2 perché la riga 1 è l'intestazione
                    stock_col = df_products.columns.get_loc('stock') + 1 
                    new_stock = df_products.at[idx, 'stock']
                    sheet_products.update_cell(sheet_row, stock_col, new_stock)
            elif status == 'out of stock':
                # Find alternative products in the same category with stock available
                product_data = collection.get(ids=[str(product_id)])
                found_products = collection.query(
                    query_texts=[f"{product_data}"],
                    n_results=5,
                    where={"category": product_data['metadatas'][0]['category']},
                    include=["documents", "metadatas"]
                )
                if found_products['documents'][0]:
                    filtered_results = []
                    for i, pid in enumerate(found_products["ids"][0]):
                        metadata = found_products["metadatas"][0][i]
                        if metadata["stock"] > 0 and pid != str(product_id):
                            filtered_results.append({
                                "product_id": pid,
                                "documents": found_products["documents"][0][i],
                                "stock": metadata["stock"],
                                "price": metadata["price"],
                            })
                    for idx,product in enumerate(filtered_results):
                        alternative_products += f"""
<PRODUCT>
    <PRODUCT DATA>{product['documents']}</PRODUCT DATA>
    <PRODUCT ID>{product['product_id']}</PRODUCT ID>
    <AVAILABLE IN STOCK>{product['stock']}</AVAILABLE IN STOCK>
    <UNIT PRICE>{product['price']}</UNIT PRICE>
</PRODUCT>"""
        
        # If the order was impossible to identify, we try to find alternative products
        if alternative_products == "" and not order_data:
            found_products = collection.query(
                query_texts=[f"{email_subject} - {email_message}"],
                n_results=5,
                include=["documents", "metadatas"]
            )
            for idx in range(len(found_products["documents"][0])):
                document = found_products["documents"][0][idx]
                metadata = found_products["metadatas"][0][idx]
    
                alternative_products += f"""
<PRODUCT>
    <PRODUCT DATA>{document}</PRODUCT DATA>
    <PRODUCT ID>{metadata.get('product_id')}</PRODUCT ID>
    <AVAILABLE IN STOCK>{metadata.get('stock')}</AVAILABLE IN STOCK>
    <UNIT PRICE>{metadata.get('price')}</UNIT PRICE>
</PRODUCT>"""

        # Generate response for the email
        response = generate_order_response(
            email_data=row,
            order_data=order_status,
            relevant_products=alternative_products,
            openai_client=openai_client
        )
        sheet_order_response.append_row([email_id, response])
    
    #Handling inquiries
    inquiries = df_email_merge[df_email_merge['category'] == 'inquiry']

    # Load order status data to identify already processed orders
    df_inquiry_response = load_data(spreadsheet, "inquiry-response")

    # Merge inquiries with inquiry response to find pending inquiries
    inquiries = inquiries.merge(
        df_inquiry_response, left_on='email_id', right_on='email ID', how='left'
    )

    # Filter out inquiries that have already been processed
    pending_inquiries = inquiries[inquiries['response'].isnull()]
    
    for idx, row in pending_inquiries.iterrows():
        email_id = row['email_id']
        email_subject = row['subject']
        email_message = row['message']

        logging.info(f"####Processing inquiry for email ID {email_id}...")

        product_data = collection.query(
            query_texts=[f"{email_subject} - {email_message}"],
            n_results=5,
            include=["documents", "metadatas"]
        )

        response = generate_inquiry_response(row, product_data, openai_client)
        sheet_inquiry_response.append_row([email_id, response])

        logging.info(f"######Inquiry response generated for email ID {email_id}...")

    logging.info("All emails processed successfully.")

if __name__ == "__main__":
    main()
