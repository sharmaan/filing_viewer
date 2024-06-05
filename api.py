import yfinance as yf
from pymongo import MongoClient
from yahooquery import Ticker
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

'''
Rest API for CRUD Operations, Visualization
'''

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return '<h1>Hello, World </h1>'


@app.route('/getData')
def get_data():
    """
          Get data for home page
       """
    db = getDB()
    stock_metadata = [s for s in db["stock_metadata"].find({}, {"_id": 0})]
    stock_financial_info = [s for s in db["stock_financial_info"].find({}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
         "currentPrice": sf["currentPrice"], "totalRevenue": sf["totalRevenue"]
            , "enterpriseValue": sd["enterpriseValue"], "beta": sd["beta"], "bookValue": sd["bookValue"],
         "priceToBook": sd["priceToBook"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info if sm["symbol"] == sf["symbol"]
        for sd in stock_default_key_stats if sm["symbol"] == sd["symbol"]
    ]

    return jsonify(result)

@app.route('/getData/<string:symbol>')
def get_data_by_trading_symbol(symbol: str):
  db = getDB()
  stock_metadata = [s for s in db["stock_metadata"].find({"symbol":symbol}, {"_id": 0})]
  stock_financial_info = [s for s in db["stock_financial_info"].find({"symbol":symbol}, {"_id": 0})]
  stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({"symbol":symbol}, {"_id": 0})]
  result = [
    {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
     "country": sm["country"], "currency": sm["currency"]
      , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
     "currentPrice": sf["currentPrice"], "totalRevenue": sf["totalRevenue"]
      , "enterpriseValue": sd["enterpriseValue"], "beta": sd["beta"], "bookValue": sd["bookValue"],
     "priceToBook": sd["priceToBook"]
     }
    for sm in stock_metadata
    for sf in stock_financial_info if sm["symbol"] == sf["symbol"]
    for sd in stock_default_key_stats if sm["symbol"] == sd["symbol"]
  ]

  return jsonify(result)

@app.route('/get_marketCap_over_1T')
def get_marketCap_Over_1T():
    """
       Get companies with marketCap over $1 Trillion Dollors
    """
    db = getDB()
    stock_metadata = [s for s in db["stock_metadata"].find({}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({}, {"_id": 0})]
    stock_financial_info = [sf for sf in
                            db["stock_financial_info"].find({"marketCap": {"$gt": 1000000000000}}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
         "currentPrice": sf["currentPrice"], "totalRevenue": sf["totalRevenue"]
            , "enterpriseValue": sd["enterpriseValue"], "beta": sd["beta"], "bookValue": sd["bookValue"],
         "priceToBook": sd["priceToBook"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info
        for sd in stock_default_key_stats
        if sm["symbol"] == sf["symbol"] and sm["symbol"] == sd["symbol"]
    ]
    return jsonify(result)


@app.route('/get_pricing_history_highest_open')
def get_pricing_history_highest_open():
    db = getDB()
    """
       Get Highesr opening price
    """
    stock_history_data = [sh for sh in db["stock_history_data"].aggregate(
        [{"$group": {"_id": "$symbol", "maxOpenHigh": {"$max": "$High"}}}])]
    stock_metadata = [s for s in db["stock_metadata"].find(
        {"symbol": {"$in": ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_financial_info = [s for s in db["stock_financial_info"].find(
        {"symbol": {"$in": ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find(
        {"symbol": {"$in": ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
         "totalRevenue": sf["totalRevenue"],
         "priceToBook": sd["priceToBook"]
            , "maxOpenHigh": sh["maxOpenHigh"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info
        for sd in stock_default_key_stats
        for sh in stock_history_data
        if sm["symbol"] == sf["symbol"] and sm["symbol"] == sd["symbol"] and sm["symbol"] == sh["_id"]
    ]
    return jsonify(result)


@app.route('/get_all_pricing_history')
def get_all_pricing_history():
    db = getDB()
    '''
    All Pricing History
    '''
    stock_history_data = [sh for sh in db["stock_history_data"].find({}, {"_id": 0}).sort(
        {"High": -1, "Date": -1, "symbol": 1, "High": -1})]
    stock_metadata = [s for s in db["stock_metadata"].find(
        {"symbol": {"$in": ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_financial_info = [s for s in db["stock_financial_info"].find(
        {"symbol": {"$in": ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find(
        {"symbol": {"$in": ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
         "priceToBook": sd["priceToBook"]
            , "Date": sh["Date"], "Open": sh["Open"], "High": sh["High"], "Low": sh["Low"], "Close": sh["Close"],
         "Volume": sh["Volume"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info
        for sd in stock_default_key_stats
        for sh in stock_history_data
        if sm["symbol"] == sf["symbol"] and sm["symbol"] == sd["symbol"] and sm["symbol"] == sh["symbol"]
    ]
    return jsonify(result)


@app.route('/get_all_pricing_history/<string:symbol>')
def get_all_pricing_history_by_symbol(symbol: str):
    """
       All Pricing History by trading symbol order by High, Date, Symbol
    """
    db = getDB()
    stock_history_data = [sh for sh in db["stock_history_data"].find({"symbol": symbol}, {"_id": 0}).sort(
        {"High": -1, "Date": -1, "symbol": 1})]
    #  db.stock_history_data.aggregate([{$group:{_id:"$symbol",maxOpenHigh:{$max:"$Open"},Date:{$max:"$Date"},Volume:{$max:"$Volume"}}}])
    stock_metadata = [s for s in db["stock_metadata"].find({"symbol": symbol}, {"_id": 0})]
    stock_financial_info = [s for s in db["stock_financial_info"].find({"symbol": symbol}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({"symbol": symbol}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
         "priceToBook": sd["priceToBook"]
            , "Date": sh["Date"], "Open": sh["Open"], "High": sh["High"], "Low": sh["Low"], "Close": sh["Close"],
         "Volume": sh["Volume"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info
        for sd in stock_default_key_stats
        for sh in stock_history_data
        if sm["symbol"] == sf["symbol"] and sm["symbol"] == sd["symbol"] and sm["symbol"] == sh["symbol"]
    ]
    return jsonify(result)


@app.route('/get_data_to_visualize')
def getDataToDisplay():
    """
         API to get data for Visualization - symbol, High Open Price
      """
    db = getDB()
    stock_history_data = [sh for sh in db["stock_history_data"].aggregate(
        [{"$group": {"_id": "$symbol", "maxOpenHigh": {"$max": "$High"}}}])]
    result = [{"symbol": d["_id"], "High": format(d["maxOpenHigh"], ".2f")} for d in stock_history_data]
    return jsonify(result)

@app.route('/get_predictions')
def get_predictions():
    """
            API to get prediction for next day closing price
    """
    # Fetch stock data
    data = yf.download('AAPL', start='2010-01-01', end='2020-01-01')
    data['Prev Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)

    # Prepare data for prediction
    X = data[['Prev Close']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)

    # Make a prediction for the next day
    prediction = model.predict([[data.iloc[-1]['Close']]])

    # Return the prediction
    return jsonify({'prediction': prediction[0]})


@app.route('/updateData/<string:symbol>', methods=['PUT'])
def updateFinanceData(symbol: str):
    """
              API to update existing data for trading symbol
      """
    data = request.get_json()
    db = getDB()
    stock_metadata = db["stock_metadata"]
    stock_financial_info = db["stock_financial_info"]
    stock_default_key_stats = db["stock_default_key_stats"]
    if (stock_metadata.find_one({"symbol": symbol})) is not None:
        stock_metadata.update_one({"symbol": symbol},
                                  {"$set": {"companyName": data["companyName"], "sector": data["sector"],
                                            "industry": data["industry"]
                                      , "country": data["country"], "currency": data["currency"]}})
        stock_financial_info.update_one({"symbol": symbol},
                                        {"$set": {"marketCap": data["marketCap"],
                                                  "sharesOutstanding": data["sharesOutstanding"],
                                                  "currentPrice": data["currentPrice"],
                                                  "totalRevenue": data["totalRevenue"]}})
        stock_default_key_stats.update_one({"symbol": symbol},
                                           {"$set": {"enterpriseValue": data["enterpriseValue"], "beta": data["beta"],
                                                     "bookValue": data["bookValue"],
                                                     "priceToBook": data["priceToBook"]}})
    return get_data()  # jsonify([a for a in getDB()["test1"].find({}, {"_id": 0})])  # get_data()


@app.route('/addData', methods=['POST'])
def addFinanceData():
    """
               API to add new data
       """
    data = request.get_json()
    db = getDB()
    stock_metadata = db["stock_metadata"]
    stock_financial_info = db["stock_financial_info"]
    stock_default_key_stats = db["stock_default_key_stats"]
    stock_metadata.insert_one({"symbol": data["symbol"], "companyName": data["companyName"], "sector": data["sector"],
                               "industry": data["industry"], "country": data["country"], "currency": data["currency"]})
    stock_financial_info.insert_one(
        {"symbol": data["symbol"], "marketCap": data["marketCap"], "sharesOutstanding": data["sharesOutstanding"],
         "currentPrice": data["currentPrice"], "totalRevenue": data["totalRevenue"]})
    stock_default_key_stats.insert_one(
        {"symbol": data["symbol"], "enterpriseValue": data["enterpriseValue"], "beta": data["beta"],
         "bookValue": data["bookValue"], "priceToBook": data["priceToBook"]})
    return get_data()


@app.route('/deleteData/<string:symbol>', methods=['DELETE'])
def deleteFinanceData(symbol: str):
    """
                  API to delete record for given symbol
    """
    db = getDB()
    stock_metadata = db["stock_metadata"]
    stock_financial_info = db["stock_financial_info"]
    stock_default_key_stats = db["stock_default_key_stats"]
    stock_metadata.delete_one({"symbol": symbol})
    stock_financial_info.delete_one({"symbol": symbol})
    stock_default_key_stats.delete_one({"symbol": symbol})
    return get_data()


def getDB():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['finance_db_infs740']
    return db


def load_ticker_history():
    db = getDB()
    collection = db['stock_history_data']
    collection.delete_many({})
    list_symbols = ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]
    for tickerSymbol in list_symbols:
        tickerData = yf.Ticker(tickerSymbol)
        # tickerDf = tickerData.history(period='1d', start='2021-1-1', end='2024-5-1')
        tickerDf = tickerData.history(period='1d', start='2023-1-1', end='2024-5-1')
        data_dict = tickerDf.reset_index().to_dict("records")
        for d in data_dict:
            d['symbol'] = tickerSymbol
        collection.insert_many(data_dict)
    print("Financial data loaded into MongoDB successfully.")
    print([f for f in collection.find({"symbol": "AAPL"})])


def load_ticker_data():
    db = getDB()
    stock_metadata = db['stock_metadata']
    stock_financial_info = db['stock_financial_info']
    stock_default_key_stats = db['stock_default_key_stats']
    ##clean up first
    stock_metadata.delete_many({})
    stock_financial_info.delete_many({})
    stock_default_key_stats.delete_many({})
    # Insert into MongoDB
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500 = tables[0]['Symbol'].tolist()
    list_ticker_symbol = [symbol.replace(".", "-") for symbol in sp500]
    N = 50
    list1 = list_ticker_symbol[:N]
    list2 = ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]
    result = list1 + [data for data in list2 if data not in list1]
    result.sort()

    for ticker_symbol in list_ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)
        metadata = ticker.info
        stock_metadata.insert_one(get_selected_metadata(metadata))
        stock_financial_info.insert_one(get_selected_financial_info(metadata))
        stock_default_key_stats.insert_one(get_selected_key_stats(metadata))

    print(f"Inserted data")


def get_selected_metadata(metadata):
    selected_metadata = {
        "symbol": metadata["symbol"],
        "companyName": metadata.get("longName"),
        "sector": metadata.get("sector"),
        "industry": metadata.get("industry"),
        "country": metadata.get("country"),
        "currency": metadata.get("currency"),
    }
    return selected_metadata


def get_selected_key_stats(metadata):
    selected_metadata = {
        "symbol": metadata.get("symbol"),
        "enterpriseValue": metadata.get("enterpriseValue"),
        "beta": metadata.get("beta"),
        "bookValue": metadata.get("bookValue"),
        "priceToBook": metadata.get("priceToBook"),
    }
    return selected_metadata


def get_selected_financial_info(metadata):
    selected_metadata = {
        "symbol": metadata.get("symbol"),
        "marketCap": metadata.get("marketCap"),
        "sharesOutstanding": metadata.get("sharesOutstanding"),
        "currentPrice": metadata.get("currentPrice"),
        "totalRevenue": metadata.get("totalRevenue")
    }
    return selected_metadata


if __name__ == '__main__':
    app.run(debug=True)
