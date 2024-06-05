import os
import time
from datetime import datetime
from operator import itemgetter

import yfinance as yf
from pymongo import MongoClient
from yahooquery import Ticker
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

import xml.etree.ElementTree as ET
import pandas as pd


'''
keep for reference
    # Fetch financial data from Yahoo Finance
    # ticker_symbol = "AAPL"
    # ticker = yf.Ticker(ticker_symbol)
    # data = ticker.history(period="max")
    # data.reset_index(inplace=True)
    # data_dict = data.to_dict("records")
    #
    # # Insert data into MongoDB
    # client = MongoClient('mongodb://localhost:27017/')
    # db = client['yahoo_finance']
    # collection = db['historical_data']
    # collection.insert_many(data_dict)
    # Fetch data from Yahoo Finance
    # tickerSymbol = 'AAPL'
    # tickerData = yf.Ticker(tickerSymbol)
    # tickerDf = tickerData.history(period='1d', start='2023-1-1', end='2024-1-1')
'''

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return '<h1>Hello, World </h1>'


@app.route('/getData')
def get_data():
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


@app.route('/get_marketCap_over_1T')
def get_marketCap_Over_1T():
    #marketCap > 1T http://127.0.0.1:5000/get_marketCap_Over_1T
    db = getDB()
    stock_metadata = [s for s in db["stock_metadata"].find({}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({}, {"_id": 0})]
    stock_financial_info =[sf for sf in db["stock_financial_info"].find({"marketCap":{"$gt":1000000000000}},{"_id":0})]
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
    db=getDB()
    '''
    max high open, aggregate query , group by then combine with metadata , financial info, default stats for more data
    '''
    # stock_history_data =[sh for sh in db["stock_history_data"].aggregate([{"$group":{"_id":"$symbol","maxOpenHigh":{"$max":"$Open"},"Date":{"$max":"$Date"},"Volume":{"$max":"$Volume"}}}])]
    stock_history_data =[sh for sh in db["stock_history_data"].aggregate([{"$group":{"_id":"$symbol","maxOpenHigh":{"$max":"$High"}}}])]
    #  db.stock_history_data.aggregate([{$group:{_id:"$symbol",maxOpenHigh:{$max:"$Open"},Date:{$max:"$Date"},Volume:{$max:"$Volume"}}}])
    stock_metadata = [s for s in db["stock_metadata"].find({"symbol":{"$in":["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_financial_info = [s for s in db["stock_financial_info"].find({"symbol":{"$in":["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({"symbol":{"$in":["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],"totalRevenue":sf["totalRevenue"],
          "priceToBook": sd["priceToBook"]
         ,"maxOpenHigh":sh["maxOpenHigh"]#,"maxOpenDate":sh["Date"]#,"maxOpenVolume":sh["Volume"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info
        for sd in stock_default_key_stats
        for sh in stock_history_data
        if sm["symbol"] == sf["symbol"] and sm["symbol"] == sd["symbol"] and sm["symbol"]==sh["_id"]
    ]
    return jsonify(result)

@app.route('/get_all_pricing_history')
def get_all_pricing_history():
    db=getDB()
    '''
    max high open, aggregate query , group by then combine with metadata , financial info, default stats for more data
    '''
    stock_history_data =[sh for sh in db["stock_history_data"].find({},{"_id":0}).sort({"High":-1,"Date":-1,"symbol":1,"High":-1})]
    #  db.stock_history_data.aggregate([{$group:{_id:"$symbol",maxOpenHigh:{$max:"$Open"},Date:{$max:"$Date"},Volume:{$max:"$Volume"}}}])
    stock_metadata = [s for s in db["stock_metadata"].find({"symbol":{"$in":["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_financial_info = [s for s in db["stock_financial_info"].find({"symbol":{"$in":["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    stock_default_key_stats = [s for s in db["stock_default_key_stats"].find({"symbol":{"$in":["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]}}, {"_id": 0})]
    result = [
        {"symbol": sm["symbol"], "companyName": sm["companyName"], "sector": sm["sector"], "industry": sm["industry"],
         "country": sm["country"], "currency": sm["currency"]
            , "marketCap": sf["marketCap"], "sharesOutstanding": sf["sharesOutstanding"],
          "priceToBook": sd["priceToBook"]
          ,"Date":sh["Date"], "Open":sh["Open"] ,"High":sh["High"],"Low":sh["Low"],"Close":sh["Close"],"Volume":sh["Volume"]
         }
        for sm in stock_metadata
        for sf in stock_financial_info
        for sd in stock_default_key_stats
        for sh in stock_history_data
        if sm["symbol"] == sf["symbol"] and sm["symbol"] == sd["symbol"] and sm["symbol"]==sh["symbol"]
    ]
    return jsonify(result)

@app.route('/get_all_pricing_history/<string:symbol>')
def get_all_pricing_history_by_symbol(symbol: str):
    db = getDB()
    '''
    max high open, aggregate query , group by then combine with metadata , financial info, default stats for more data
    '''
    stock_history_data = [sh for sh in db["stock_history_data"].find({"symbol":symbol}, {"_id": 0}).sort(
      {"High": -1, "Date": -1, "symbol": 1, "High": -1})]
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
  db=getDB()
  stock_history_data = [sh for sh in db["stock_history_data"].aggregate([{"$group": {"_id": "$symbol", "maxOpenHigh": {"$max": "$High"}}}])]
  result = [{"symbol":d["_id"],"High":format(d["maxOpenHigh"],".2f") } for d in stock_history_data]
  return jsonify(result)


@app.route('/get_data_to_visualize_scatter')
def getDataToDisplay_Scatter():
  db=getDB()
  stock_history_data = [sh for sh in db["stock_history_data"].aggregate([{"$group": {"_id": "$symbol", "maxOpenHigh": {"$max": "$High"}}}])]
  stock_history_data = [sh for sh in db["stock_history_data"].find({}, {"_id": 0}).sort( {"High": -1, "Date": -1, "symbol": 1, "High": -1})]
  result = [{"symbol":d["symbol"],"price":format(d["High"],".2f"),"date":d["Date"] } for d in stock_history_data]
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

@app.route('/get_predictions')
def get_predictions():
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
    print(f"symbol {symbol}")
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
                                                "sharesOutstanding": data["sharesOutstanding"],"currentPrice":data["currentPrice"],"totalRevenue":data["totalRevenue"]}})
        stock_default_key_stats.update_one({"symbol": symbol},
                                           {"$set": {"enterpriseValue": data["enterpriseValue"],"beta":data["beta"],"bookValue":data["bookValue"],"priceToBook":data["priceToBook"]}})
    return get_data()  # jsonify([a for a in getDB()["test1"].find({}, {"_id": 0})])  # get_data()


@app.route('/addData', methods=['POST'])
def addFinanceData():
    data = request.get_json()
    db = getDB()
    stock_metadata = db["stock_metadata"]
    stock_financial_info = db["stock_financial_info"]
    stock_default_key_stats = db["stock_default_key_stats"]
    stock_metadata.insert_one({"symbol": data["symbol"], "companyName": data["companyName"], "sector": data["sector"],
                               "industry": data["industry"] , "country": data["country"], "currency": data["currency"]})
    stock_financial_info.insert_one(
        {"symbol": data["symbol"], "marketCap": data["marketCap"], "sharesOutstanding": data["sharesOutstanding"],"currentPrice":data["currentPrice"],"totalRevenue":data["totalRevenue"]})
    stock_default_key_stats.insert_one(
        {"symbol": data["symbol"], "enterpriseValue": data["enterpriseValue"],"beta":data["beta"],"bookValue":data["bookValue"],"priceToBook":data["priceToBook"]})
    # return jsonify({"message": "Data inserted", "document_id": str(result.inserted_id)})
    return get_data()


@app.route('/deleteData/<string:symbol>', methods=['DELETE'])
def deleteFinanceData(symbol: str):
    print(f"symbol {symbol}")
    # if getDB()["test1"].find_one({"symbol": symbol}) is not None:
    #     print(f"found {symbol} to delete")
    #     result = getDB()["test1"].delete_one({"symbol": symbol})
    # return jsonify(
    #     {"message": "Data deleted", "symbol": symbol})  # jsonify([a for a in getDB()["test1"].find({},{"_id":0})])
    db = getDB()
    stock_metadata = db["stock_metadata"]
    stock_financial_info = db["stock_financial_info"]
    stock_default_key_stats = db["stock_default_key_stats"]
    stock_metadata.delete_one({"symbol": symbol})
    stock_financial_info.delete_one({"symbol": symbol})
    stock_default_key_stats.delete_one({"symbol": symbol})
    return get_data()


@app.route('/url_variables/<string:name>/<int:age>')
def url_variables(name: str, age: int):
    if age < 18:
        return jsonify(message="Sorry " + name + ", you are not old enough."), 401
    else:
        return jsonify(message="Welcome " + name + ", you are old enough!")


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
    N=50
    list1 = list_ticker_symbol[:N]
    list2 = ["GOOGL", "GOOG", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]
    result = list1 + [data for data in list2 if data not in list1]
    result.sort()
    print(list1)
    print(list2)
    print(result)

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


def load_ticker_asset_profile():
    list_ticker_symbol = ['AAPL', 'MSFT']
    list_ticker_symbol = ['AAPL']
    for ticker_symbol in list_ticker_symbol:
        ticker = Ticker(ticker_symbol)
        ##fetch basic info
        info = ticker.asset_profile
        print(f"ticker_asset_profile {info[ticker_symbol]}")
        ##fetch historical market data (ex. for the past month)
        # hist = ticker.history(period='1mo')
        # print(f"History {hist}")


def load_ticker_calender_events():
    list_ticker_symbol = ['AAPL', 'MSFT']
    # list_ticker_symbol = ['AAPL']
    db = getDB()
    collection = db["stock_calendar_events"]
    collection.delete_many({})
    for ticker_symbol in list_ticker_symbol:
        ticker = Ticker(ticker_symbol)
        ##fetch basic info
        info = ticker.calendar_events[ticker_symbol]['earnings']
        info['symbol'] = ticker_symbol
        info["event_date"] = info['earningsDate'][0][:10]
        # print(info)
        result = collection.insert_one(info)
        print(f"Inserted metadata with id: {result.inserted_id}")
    print("***********************************************")
    print([t for t in collection.find({"symbol": "AAPL"})])


# def load_ticker_history():
#     list_ticker_symbol = ['AAPL', 'MSFT']
#     # list_ticker_symbol = ['AAPL']
#     db = getDB()
#     collection = db["stock_history"]
#     collection.delete_many({})
#     print(f"deleted collection")
#     for ticker_symbol in list_ticker_symbol:
#         ticker = Ticker(ticker_symbol)
#         ##fetch basic info
#         history_1mo = ticker.history(period='1mo')
#         list_dict = pd.DataFrame(history_1mo).to_dict('records')
#         for d in list_dict:
#             d['symbol'] = ticker_symbol
#         collection.insert_many(list_dict)
#     print(f"Inserted {list_ticker_symbol}")
#     print("***********************************************")
#     print([t for t in collection.find({"symbol": "AAPL"})])


def load_ticker_key_stats():
    list_ticker_symbol = ['AAPL', 'MSFT']
    # list_ticker_symbol = ['AAPL']
    db = getDB()
    collection = db["stock_default_key_stats"]
    collection.delete_many({})
    print(f"deleted collection key_stats")
    for ticker_symbol in list_ticker_symbol:
        ticker = Ticker(ticker_symbol)
        ##fetch basic info
        ticker_key_stats = ticker.key_stats[ticker_symbol]
        ticker_key_stats['symbol'] = ticker_symbol
        selected_stats = get_selected_key_stats(ticker_key_stats)
        # print(f"ticker_key_stats{ticker_key_stats}")
        collection.insert_one(selected_stats)

    print([k for k in collection.find({"symbol": "AAPL"})])


# load_ticker_asset_profile()
# load_ticker_calender_events()
# load_ticker_history_one_month()
# load_ticker_key_stats()
# ticker = Ticker('AAPL')
# print( ticker.history(period='1mo'))
# df = pd.DataFrame(ticker.history(period='1mo'))
# print(df.to_dict().keys())
# getDB()["test"].insert_one(ticker.history(period='1mo'))
# print( Ticker('AAPL').calendar_events)

# Assuming the structure of calendar_events is as expected
# Extract earnings date and ex-dividend date
# Define the ticker symbol
# print( [a for a in getDB()["stock_metadata"].find({},{"_id":0})])
def check():
    # db = getDB()
    # stock_metadata = db["stock_metadata"]
    # print([sm for sm in stock_metadata.find({},{"_id":0}) ])
    # symbols=['fb','aapl','msft','amzn','tsla','nflx','goog']
    # symbols=['aapl']
    # YF_aapl=yf.Ticker('aapl')
    # YQ_aapl = Ticker('aapl')
    # attrs = ['cashflow', 'balance_sheet', 'financials']
    # yf_d = {}
    # for attr in attrs:
    #     df = getattr(YF_aapl, attr)
    #     yf_d[attr] = df.shape[0]

    # print(tickers.earnings)
    db = getDB()
    collection = db['stock_metadata_all']
    collection.delete_many({})
    list_ticker_symbol = ['AAPL', 'MSFT']
    for ticker_symbol in list_ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)
        metadata = ticker.info
        selected_metadata = get_selected_metadata(metadata)
        result = collection.insert_one(metadata)
        print(f"Inserted metadata with id: {result.inserted_id}")


def sp500_historical_pricing():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500 = tables[0]['Symbol'].tolist()
    sp500 = [symbol.replace(".", "-") for symbol in sp500]
    print(f"sp500", sp500)
    sp500 = ['aapl']
    # yf_data = yf.download(sp500, period='ytd', interval='1d', group_by='ticker')
    yf_data = yf.download(sp500, period='ytd', interval='1d', group_by='ticker')
    type(yf_data)
    print(yf_data.shape)
    print(yf_data.head())

    t = Ticker('aapl')
    keys = t.all_modules['aapl'].keys()
    # print( keys)
    count = 0
    for key in keys:
        print(t.all_modules['aapl'][key])
        count = count + 1
        if count > 1:
            break


def get_date_from_timestamp():
    global timestamp
    timestamp = 1696032000
    date = datetime.fromtimestamp(timestamp)
    print(f"lastFiscalYearEnd {date}")
    timestamp = 1727654400
    date = datetime.fromtimestamp(timestamp)
    print(f"nextFiscalYearEnd {date}")

import requests
from bs4 import BeautifulSoup
import pandas as pd
def extract_data():
    # Fetch the XML content from the URL
    url = "https://www.sec.gov/Archives/edgar/xbrl-inline.rss.xml"
    response = requests.get(url)
    xml_content = response.content
    print(xml_content)
    # Parse the XML content using BeautifulSoup
    soup = BeautifulSoup(xml_content, "xml")

    # Extract information between <item> tags
    items = soup.find_all("item")
    item_details = []

    for item in items:
        item_detail = {
            "title": item.find("title").text if item.find("title") else None,
            "link": item.find("link").text if item.find("link") else None,
            "description": item.find("description").text if item.find("description") else None,
            "pubDate": item.find("pubDate").text if item.find("pubDate") else None
        }
        print(item)
        item_details.append(item_detail)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(item_details)

    # Display the DataFrame
    print(df)
import xml.dom.minidom
def extract_data_local():
  doc=xml.dom.minidom.parse("xbrl-inline.rss.xml")
  print(doc.nodeName)
  print(doc.firstChild.tagName)
  xbrl_filings = doc.getElementsByTagName("edgar:xbrlFiling")
  print("%d xbrl filings:" %xbrl_filings.length)
  for xbrl_filing in xbrl_filings:
    print(xbrl_filing.getElementsByTagName("edgar:companyName"))

def extract_data_local1():
  import xml.etree.ElementTree as ET
  import pandas as pd

  # Load the XML file
  tree = ET.parse('xbrl-inline.rss.xml')
  root = tree.getroot()

  # Define namespaces to handle the XML tags with namespaces
  namespaces = {
    'edgar': 'https://www.sec.gov/Archives/edgar'
  }

  # Extract data from <item> and <edgar:xbrlFiling> tags
  data = []
  for item in root.findall('channel/item'):
    item_data = {
      'title': item.find('title').text if item.find('title') is not None else None,
      'link': item.find('link').text if item.find('link') is not None else None,
      'description': item.find('description').text if item.find('description') is not None else None,
      'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else None
    }

    xbrlFiling = item.find('edgar:xbrlFiling', namespaces)
    if xbrlFiling is not None:
      item_data.update({
        'companyName': xbrlFiling.find('edgar:companyName', namespaces).text if xbrlFiling.find('edgar:companyName',
                                                                                                namespaces) is not None else None,
        'formType': xbrlFiling.find('edgar:formType', namespaces).text if xbrlFiling.find('edgar:formType',
                                                                                          namespaces) is not None else None,
        'filingDate': xbrlFiling.find('edgar:filingDate', namespaces).text if xbrlFiling.find('edgar:filingDate',
                                                                                              namespaces) is not None else None,
        'cikNumber': xbrlFiling.find('edgar:cikNumber', namespaces).text if xbrlFiling.find('edgar:cikNumber',
                                                                                            namespaces) is not None else None,
        'accessionNumber': xbrlFiling.find('edgar:accessionNumber', namespaces).text if xbrlFiling.find(
          'edgar:accessionNumber', namespaces) is not None else None,
        'fileNumber': xbrlFiling.find('edgar:fileNumber', namespaces).text if xbrlFiling.find('edgar:fileNumber',
                                                                                              namespaces) is not None else None,
        'acceptanceDatetime': xbrlFiling.find('edgar:acceptanceDatetime', namespaces).text if xbrlFiling.find(
          'edgar:acceptanceDatetime', namespaces) is not None else None,
        'period': xbrlFiling.find('edgar:period', namespaces).text if xbrlFiling.find('edgar:period',
                                                                                      namespaces) is not None else None,
        'fiscalYearEnd': xbrlFiling.find('edgar:fiscalYearEnd', namespaces).text if xbrlFiling.find(
          'edgar:fiscalYearEnd', namespaces) is not None else None
      })

    data.append(item_data)

  # Convert the extracted data to a DataFrame
  df = pd.DataFrame(data)

  # Save the DataFrame to a CSV file and display the first few rows
  df.to_csv('extracted_sec_rss_feed_data.csv', index=False)
  df.head()
  print(df.head())

def extract_data_local2():
  import xml.etree.ElementTree as ET
  import pandas as pd

  # Load the XML file
  tree = ET.parse('xbrl-inline.rss.xml')
  root = tree.getroot()

  # Define namespaces to handle the XML tags with namespaces
  namespaces = {
    'edgar': 'https://www.sec.gov/Archives/edgar'
  }

  # Extract data from <item> and <edgar:xbrlFiling> tags
  data = []
  for item in root.findall('channel/item'):
    item_data = {
      'title': item.find('title').text if item.find('title') is not None else None,
      'link': item.find('link').text if item.find('link') is not None else None,
      'description': item.find('description').text if item.find('description') is not None else None,
      'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else None
    }

    xbrlFiling = item.find('edgar:xbrlFiling', namespaces)
    if xbrlFiling is not None:
      item_data.update({
        'companyName': xbrlFiling.find('edgar:companyName', namespaces).text if xbrlFiling.find('edgar:companyName',
                                                                                                namespaces) is not None else None,
        'formType': xbrlFiling.find('edgar:formType', namespaces).text if xbrlFiling.find('edgar:formType',
                                                                                          namespaces) is not None else None,
        'filingDate': xbrlFiling.find('edgar:filingDate', namespaces).text if xbrlFiling.find('edgar:filingDate',
                                                                                              namespaces) is not None else None,
        'cikNumber': xbrlFiling.find('edgar:cikNumber', namespaces).text if xbrlFiling.find('edgar:cikNumber',
                                                                                            namespaces) is not None else None,
        'accessionNumber': xbrlFiling.find('edgar:accessionNumber', namespaces).text if xbrlFiling.find(
          'edgar:accessionNumber', namespaces) is not None else None,
        'fileNumber': xbrlFiling.find('edgar:fileNumber', namespaces).text if xbrlFiling.find('edgar:fileNumber',
                                                                                              namespaces) is not None else None,
        'acceptanceDatetime': xbrlFiling.find('edgar:acceptanceDatetime', namespaces).text if xbrlFiling.find(
          'edgar:acceptanceDatetime', namespaces) is not None else None,
        'period': xbrlFiling.find('edgar:period', namespaces).text if xbrlFiling.find('edgar:period',
                                                                                      namespaces) is not None else None,
        'fiscalYearEnd': xbrlFiling.find('edgar:fiscalYearEnd', namespaces).text if xbrlFiling.find(
          'edgar:fiscalYearEnd', namespaces) is not None else None
      })

      # Extract xbrlFiles
      xbrlFiles = []
      for xbrlFile in xbrlFiling.findall('edgar:xbrlFile', namespaces):
        xbrlFiles.append({
          'sequence': xbrlFile.get('sequence'),
          'file': xbrlFile.get('file'),
          'type': xbrlFile.get('type'),
          'size': xbrlFile.get('size'),
          'description': xbrlFile.get('description'),
          'url': xbrlFile.get('url')
        })
      item_data['xbrlFiles'] = xbrlFiles

    data.append(item_data)

  # Convert the extracted data to a DataFrame
  df = pd.DataFrame(data)

  # Normalize the xbrlFiles column to create a separate DataFrame
  xbrl_files_df = df.explode('xbrlFiles').reset_index(drop=True)
  xbrl_files_df = pd.concat([xbrl_files_df.drop(['xbrlFiles'], axis=1), xbrl_files_df['xbrlFiles'].apply(pd.Series)],
                            axis=1)

  # Save the DataFrame to a CSV file and display the first few rows
  df.to_csv('extracted_sec_rss_feed_data.csv', index=False)
  xbrl_files_df.to_csv('extracted_sec_rss_feed_xbrl_files_data.csv', index=False)
  df.head(), xbrl_files_df.head()
  print(df.tail(), xbrl_files_df.tail())

def extract_data_local3():
  import xml.etree.ElementTree as ET
  import pandas as pd

  # Load the XML file
  tree = ET.parse('xbrl-inline.rss.xml')
  root = tree.getroot()

  # Define namespaces to handle the XML tags with namespaces
  namespaces = {
    'edgar': 'https://www.sec.gov/Archives/edgar'
  }

  # Extract data from <item> and <edgar:xbrlFiling> tags
  data = []
  for item in root.findall('channel/item'):
    item_data = {
      'title': item.find('title').text if item.find('title') is not None else None,
      'link': item.find('link').text if item.find('link') is not None else None,
      'description': item.find('description').text if item.find('description') is not None else None,
      'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else None
    }

    xbrlFiling = item.find('edgar:xbrlFiling', namespaces)
    if xbrlFiling is not None:
      item_data.update({
        'companyName': xbrlFiling.find('edgar:companyName', namespaces).text if xbrlFiling.find('edgar:companyName',
                                                                                                namespaces) is not None else None,
        'formType': xbrlFiling.find('edgar:formType', namespaces).text if xbrlFiling.find('edgar:formType',
                                                                                          namespaces) is not None else None,
        'filingDate': xbrlFiling.find('edgar:filingDate', namespaces).text if xbrlFiling.find('edgar:filingDate',
                                                                                              namespaces) is not None else None,
        'cikNumber': xbrlFiling.find('edgar:cikNumber', namespaces).text if xbrlFiling.find('edgar:cikNumber',
                                                                                            namespaces) is not None else None,
        'accessionNumber': xbrlFiling.find('edgar:accessionNumber', namespaces).text if xbrlFiling.find(
          'edgar:accessionNumber', namespaces) is not None else None,
        'fileNumber': xbrlFiling.find('edgar:fileNumber', namespaces).text if xbrlFiling.find('edgar:fileNumber',
                                                                                              namespaces) is not None else None,
        'acceptanceDatetime': xbrlFiling.find('edgar:acceptanceDatetime', namespaces).text if xbrlFiling.find(
          'edgar:acceptanceDatetime', namespaces) is not None else None,
        'period': xbrlFiling.find('edgar:period', namespaces).text if xbrlFiling.find('edgar:period',
                                                                                      namespaces) is not None else None,
        'fiscalYearEnd': xbrlFiling.find('edgar:fiscalYearEnd', namespaces).text if xbrlFiling.find(
          'edgar:fiscalYearEnd', namespaces) is not None else None
      })

      # Extract xbrlFiles
      # xbrlFiles = []
      # for xbrlFile in xbrlFiling.findall('edgar:xbrlFile', namespaces):
      #   xbrlFiles.append({
      #     'sequence': xbrlFile.get('{https://www.sec.gov/Archives/edgar}sequence'),
      #     'file': xbrlFile.get('{https://www.sec.gov/Archives/edgar}file'),
      #     'type': xbrlFile.get('{https://www.sec.gov/Archives/edgar}type'),
      #     'size': xbrlFile.get('{https://www.sec.gov/Archives/edgar}size'),
      #     'description': xbrlFile.get('{https://www.sec.gov/Archives/edgar}description'),
      #     'url': xbrlFile.get('{https://www.sec.gov/Archives/edgar}url')
      #   })
      # item_data['xbrlFiles'] = xbrlFiles

    data.append(item_data)

  # Convert the extracted data to a DataFrame
  df = pd.DataFrame(data)

  # Normalize the xbrlFiles column to create a separate DataFrame
  # xbrl_files_df = df.explode('xbrlFiles').reset_index(drop=True)
  # xbrl_files_df = pd.concat([xbrl_files_df.drop(['xbrlFiles'], axis=1), xbrl_files_df['xbrlFiles'].apply(pd.Series)],
  #                           axis=1)

  # Save the DataFrame to a CSV file and display the first few rows
  df.to_csv('extracted_sec_rss_feed_data.csv', index=False)
  # xbrl_files_df.to_csv('extracted_sec_rss_feed_xbrl_files_data.csv', index=False)
  df.head()#, xbrl_files_df.head()
  print(df.head())

def extract_data_local4():
  import xml.etree.ElementTree as ET
  import pandas as pd

  # Load the XML file
  tree = ET.parse('xbrl-inline.rss.xml')
  root = tree.getroot()

  # Define namespaces to handle the XML tags with namespaces
  namespaces = {
    'edgar': 'https://www.sec.gov/Archives/edgar'
  }

  # Extract data from <item> and <edgar:xbrlFiling> tags
  data = []
  for item in root.findall('channel/item'):
    item_data = {
      'title': item.find('title').text if item.find('title') is not None else None,
      'link': item.find('link').text if item.find('link') is not None else None,
      'description': item.find('description').text if item.find('description') is not None else None,
      'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else None
    }

    xbrlFiling = item.find('edgar:xbrlFiling', namespaces)
    if xbrlFiling is not None:
      item_data.update({
        'companyName': xbrlFiling.find('edgar:companyName', namespaces).text if xbrlFiling.find('edgar:companyName',
                                                                                                namespaces) is not None else None,
        'formType': xbrlFiling.find('edgar:formType', namespaces).text if xbrlFiling.find('edgar:formType',
                                                                                          namespaces) is not None else None,
        'filingDate': xbrlFiling.find('edgar:filingDate', namespaces).text if xbrlFiling.find('edgar:filingDate',
                                                                                              namespaces) is not None else None,
        'cikNumber': xbrlFiling.find('edgar:cikNumber', namespaces).text if xbrlFiling.find('edgar:cikNumber',
                                                                                            namespaces) is not None else None,
        'accessionNumber': xbrlFiling.find('edgar:accessionNumber', namespaces).text if xbrlFiling.find(
          'edgar:accessionNumber', namespaces) is not None else None,
        'fileNumber': xbrlFiling.find('edgar:fileNumber', namespaces).text if xbrlFiling.find('edgar:fileNumber',
                                                                                              namespaces) is not None else None,
        'acceptanceDatetime': xbrlFiling.find('edgar:acceptanceDatetime', namespaces).text if xbrlFiling.find(
          'edgar:acceptanceDatetime', namespaces) is not None else None,
        'period': xbrlFiling.find('edgar:period', namespaces).text if xbrlFiling.find('edgar:period',
                                                                                      namespaces) is not None else None,
        'fiscalYearEnd': xbrlFiling.find('edgar:fiscalYearEnd', namespaces).text if xbrlFiling.find(
          'edgar:fiscalYearEnd', namespaces) is not None else None
      })

      # Extract xbrlFiles
      xbrlFiles = []
      for xbrlFile in xbrlFiling.findall('edgar:xbrlFile', namespaces):
        xbrlFiles.append({
          'sequence': xbrlFile.get('sequence'),
          'file': xbrlFile.get('file'),
          'type': xbrlFile.get('type'),
          'size': xbrlFile.get('size'),
          'description': xbrlFile.get('description'),
          'url': xbrlFile.get('url')
        })
      item_data['xbrlFiles'] = xbrlFiles if xbrlFiles else None

    data.append(item_data)

  # Convert the extracted data to a DataFrame
  df = pd.DataFrame(data)

  # Normalize the xbrlFiles column to create a separate DataFrame
  xbrl_files_df = df.explode('xbrlFiles').reset_index(drop=True)
  xbrl_files_df = pd.concat([xbrl_files_df.drop(['xbrlFiles'], axis=1), xbrl_files_df['xbrlFiles'].apply(pd.Series)],
                            axis=1)

  # Save the DataFrame to a CSV file and display the first few rows
  df.to_csv('extracted_sec_rss_feed_data.csv', index=False)
  xbrl_files_df.to_csv('extracted_sec_rss_feed_xbrl_files_data.csv', index=False)
  df.head(), xbrl_files_df.head()
  print(df.head(),xbrl_files_df.head())

def download_file_from_web():
  import requests

  print(os.getcwd())
  # URL of the file to be downloaded
  url = "https://www.sec.gov/Archives/edgar/xbrl-inline.rss.xml"

  # Set a user-agent header to mimic a browser
  headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
  }

  # Send a GET request to the URL
  response = requests.get(url,headers=headers)

  # Check if the request was successful
  if response.status_code == 200:
    # Specify the path where the file will be saved
    file_path = "xbrl-inline.rss.xml"

    # Write the content to the file
    with open(file_path, 'wb') as file:
      file.write(response.content)

    print(f"File successfully downloaded and saved to {file_path}")
  else:
    print(f"Failed to download the file. Status code: {response.status_code}")

def extract_data():
  # Load the XML file
  tree = ET.parse('xbrl-inline.rss.xml')
  root = tree.getroot()

  # Define namespaces to handle the XML tags with namespaces
  namespaces = {
    'edgar': 'https://www.sec.gov/Archives/edgar'
  }

  # Extract data from <item> and <edgar:xbrlFiling> tags
  data = []
  for item in root.findall('channel/item'):
    item_data = {
      'title': item.find('title').text if item.find('title') is not None else None,
      'link': item.find('link').text if item.find('link') is not None else None,
      'description': item.find('description').text if item.find('description') is not None else None,
      'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else None
    }

    xbrlFiling = item.find('edgar:xbrlFiling', namespaces)
    if xbrlFiling is not None:
      item_data.update({
        'companyName': xbrlFiling.find('edgar:companyName', namespaces).text if xbrlFiling.find('edgar:companyName',
                                                                                                namespaces) is not None else None,
        'formType': xbrlFiling.find('edgar:formType', namespaces).text if xbrlFiling.find('edgar:formType',
                                                                                          namespaces) is not None else None,
        'filingDate': xbrlFiling.find('edgar:filingDate', namespaces).text if xbrlFiling.find('edgar:filingDate',
                                                                                              namespaces) is not None else None,
        'cikNumber': xbrlFiling.find('edgar:cikNumber', namespaces).text if xbrlFiling.find('edgar:cikNumber',
                                                                                            namespaces) is not None else None,
        'accessionNumber': xbrlFiling.find('edgar:accessionNumber', namespaces).text if xbrlFiling.find(
          'edgar:accessionNumber', namespaces) is not None else None,
        'fileNumber': xbrlFiling.find('edgar:fileNumber', namespaces).text if xbrlFiling.find('edgar:fileNumber',
                                                                                              namespaces) is not None else None,
        'acceptanceDatetime': format_timestamp(xbrlFiling.find('edgar:acceptanceDatetime', namespaces).text) if xbrlFiling.find(
          'edgar:acceptanceDatetime', namespaces) is not None else None,
        'period': xbrlFiling.find('edgar:period', namespaces).text if xbrlFiling.find('edgar:period',
                                                                                      namespaces) is not None else None,
        'fiscalYearEnd': xbrlFiling.find('edgar:fiscalYearEnd', namespaces).text if xbrlFiling.find('edgar:fiscalYearEnd', namespaces) is not None else None,
        'assignedSic': xbrlFiling.find('edgar:assignedSic', namespaces).text if xbrlFiling.find('edgar:assignedSic', namespaces) is not None else None,
        'assistantDirector': xbrlFiling.find('edgar:assistantDirector', namespaces).text if xbrlFiling.find('edgar:assistantDirector',namespaces) is not None else None

      })



    data.append(item_data)


  df = pd.DataFrame(data)


  # df.to_csv('extracted_sec_rss_feed_data.csv', index=False)

  # df.head()
  list_of_dict=df.to_dict('records')
  # print(list_of_dict)
  return list_of_dict


@app.route('/get_xbrl_inline')
def get_xbrl_inline():
  download_file_from_web()
  data = extract_data()
  return jsonify(data)

def format_timestamp(timestamp: str) -> str:
    # Parse the input timestamp to a datetime object
    date = datetime.strptime(timestamp, '%Y%m%d%H%M%S')

    # Format the datetime object to the desired string format
    formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_date

if __name__ == '__main__':
     # load_ticker_data()
    # load_ticker_history()
    # pass
    app.run(debug=True)
    # get_date_from_timestamp()
    # extract_data_local1()
    # extract_data_local4()
    # download_file_from_web()
    # extract_data_local3()
    # extract_data()
