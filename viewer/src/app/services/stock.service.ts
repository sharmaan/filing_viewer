import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: "root",
})
export class StockService {
  constructor(private _http: HttpClient) {}

  addFinanceData(data: any): Observable<any> {
    return this._http.post("http://localhost:5000/addData", data);
  }

  updateFinanceData(symbol: string, data: any): Observable<any> {
    return this._http.put(`http://localhost:5000/updateData/${symbol}`, data);
  }

  getFinanceDataList(): Observable<any> {
    return this._http.get("http://localhost:5000/getData");
  }

  deleteFinanceData(symbol: string): Observable<any> {
    return this._http.delete(`http://localhost:5000/deleteData/${symbol}`);
  }

  getFinanceOver1T(): Observable<any> {
    return this._http.get("http://127.0.0.1:5000/get_marketCap_over_1T");
  }

  getFilingData(): Observable<any>{
    return this._http.get("http://127.0.0.1:5000/get_xbrl_inline");
  }

  getHighestOpenPricingHistory(): Observable<any> {
    return this._http.get(
      "http://127.0.0.1:5000/get_pricing_history_highest_open"
    );
  }
  getAllPricingHistory(): Observable<any> {
    return this._http.get("http://127.0.0.1:5000/get_all_pricing_history");
  }
  getDataToVisualize(): Observable<any> {
    return this._http.get("http://127.0.0.1:5000/get_data_to_visualize");
  }
  getDataToVisualize_scatter(): Observable<any> {
    return this._http.get(
      "http://127.0.0.1:5000/get_data_to_visualize_scatter"
    );
  }
}
