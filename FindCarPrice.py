import requests
from bs4 import BeautifulSoup
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
file_location = 'CurrentCarPricePredictor/car_data.csv'
# go online and find your own
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 9_9_3) AppleWebKit/536.41 (KHTML, like Gecko) Chrome/54.0.2487.289 Safari/603'

def scrape_autotrader(ids_dict):
    """
    Function to scrape car data from AutoTrader
    """
    make = ids_dict['make']
    model = ids_dict['model']
    year_range = [ids_dict['year']-3,ids_dict['year']+3] #later add validity check for current date, autotrader will work no matter what date
    odometer_range = [ids_dict['odometer']-10000,ids_dict['odometer']+10000] #later add validity check for >0, autotrader will work no matter what kms
    province = ids_dict['province']
    postalcode = ids_dict['postalcode']

    base_url = "https://www.autotrader.ca/cars/"
    search_url = f"{base_url}{make}/{model}/on/?rcp=15&rcs=0&srt=35&yRng={year_range[0]}%2C{year_range[1]}&oRng={odometer_range[0]}%2C{odometer_range[1]}&prx=-2&prv={province}&loc={postalcode}&hprc=True&wcp=True&sts=New-Used&inMarket=advancedSearch"    
    
    # confuse website that we are not scraping
    headers = {
    'User-Agent': user_agent,
    'From': 'youremail@domain.example'
}
    # print URL
    print(f"Fetching data from: {search_url}")
    
    # make the request
    response = requests.get(search_url,headers=headers)
    # print(response.text)  # check if the page content is fully loaded

    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup.prettify())  # check the full HTML

    car_data = []
    
    # scraping
    listings = soup.find_all('div', class_='dealer-split-wrapper', limit =50) 
    # print(listings)
    for listing in listings:
        print('here')
        try:
            price = listing.find('span', class_='price-amount').get_text(strip=True)
            mileage = listing.find('span', class_='odometer-proximity').get_text(strip=True)
            print(price,mileage)
            car_data.append([ mileage, price])
        except:
            continue

    # save to CSV
    with open(file_location, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Mileage', 'Price'])
        writer.writerows(car_data)

    print("Data scraped and saved to car_data.csv")

def estimate_price_from_csv(car_model, year=None, mileage=None):
    # Load the scraped data
    df = pd.read_csv(file_location)

    # Calculate the average price
    df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
    average_price = df['Price'].mean()

    if not df.empty:
        print(f"estimated average price for {car_model}: ${average_price:.2f}")
    else:
        print("no data found for car.")

def train_ml_model():
    # Load the scraped data
    df = pd.read_csv(file_location)

    # remove garbage from data
    df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
    df['Mileage'] = df['Mileage'].replace('[\sA-Za-z,]', '', regex=True).astype(float)

    # use Mileage and find Price
    X = df[['Mileage']]
    y = df['Price']

    # split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate score
    score = model.score(X_test, y_test)
    print(f"model R^2 score: {score}")

    return model

def main():
    # Example search input from user
    year, make, model,odometer,province,postalcode = input("Enter year,make,model,odometer,province (2 letter),postalcode inputs: ").split()
    
    ids_dict = {
    'make': make,
    'year': int(year),
    'model': model,
    'odometer': int(odometer),
    'province': province,
    'postalcode': postalcode
}
    # scrape 
    scrape_autotrader(ids_dict)

    # estimate with mean
    estimate_price_from_csv(model, year, odometer)

    # ML
    model = train_ml_model()

    # predicting the price for a specific car milage
    predicted_price = model.predict([[int(odometer)]])
    print(f"Predicted price: ${predicted_price[0]:.2f}")

# boilerplate to run the main function
if __name__ == "__main__":
    main()
