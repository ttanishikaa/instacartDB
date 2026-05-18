
import pandas as pd
import numpy as np
import os

def create_customer_features():
    """
    Loads raw Instacart data, engineers customer-level features,
    and saves them to a CSV file.
    """
    print("Starting feature engineering process...")
    
    # Define file paths
    data_dir = './data'
    orders_path = os.path.join(data_dir, 'orders.csv')
    order_products_path = os.path.join(data_dir, 'order_products__train.csv')
    products_path = os.path.join(data_dir, 'products.csv')
    output_path = os.path.join(data_dir, 'customer_features.csv')

    # Load datasets
    try:
        print("Loading raw data files...")
        orders = pd.read_csv(orders_path)
        order_products = pd.read_csv(order_products_path)
        products = pd.read_csv(products_path)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the 'data' directory with the original CSV files exists.")
        return

    # Customer-level features from all of their orders
    print("Aggregating customer order data...")
    customer_agg = orders.groupby('user_id').agg(
        total_orders=('order_number', 'max'),
        avg_days_between_orders=('days_since_prior_order', 'mean'),
        avg_shopping_hour=('order_hour_of_day', 'mean')
    ).reset_index()

    # Features from the 'train' set of products
    print("Calculating features from training data...")
    train_order_details = order_products.merge(orders[['order_id', 'user_id']], on='order_id')

    # Calculate average products per order
    products_per_order = train_order_details.groupby(['user_id', 'order_id']).size().reset_index(name='n_products')
    avg_products = products_per_order.groupby('user_id')['n_products'].mean().reset_index(name='avg_products_per_order')

    # Calculate department diversity
    dept_details = train_order_details.merge(products[['product_id', 'department_id']], on='product_id')
    dept_diversity = dept_details.groupby('user_id')['department_id'].nunique().reset_index(name='department_diversity')

    # Merge all features together
    print("Merging all features...")
    customer_features = customer_agg.merge(avg_products, on='user_id').merge(dept_diversity, on='user_id')

    # Handle missing values
    print("Handling missing values...")
    customer_features['avg_days_between_orders'].fillna(0, inplace=True)
    # Save to CSV
    try:
        print(f"Saving features to {output_path}...")
        customer_features.to_csv(output_path, index=False)
        print(f"Successfully created '{output_path}' with {len(customer_features)} records.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    create_customer_features()
