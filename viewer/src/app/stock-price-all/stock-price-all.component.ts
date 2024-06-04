import { AfterViewInit, Component, OnInit, ViewChild } from '@angular/core';
import { MatSort, Sort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';
import { StockService } from '../services/stock.service';
import { LiveAnnouncer } from '@angular/cdk/a11y';
import { MatPaginator } from '@angular/material/paginator';

@Component({
  selector: 'app-stock-price-all',
  templateUrl: './stock-price-all.component.html',
  styleUrl: './stock-price-all.component.scss',
})
export class StockPriceAllComponent implements OnInit, AfterViewInit {
  ngOnInit(): void {
    this.getData();
  }
  selected = true;
  displayedColumns: string[] = [
    'Date', //stock_history_data
    'symbol', //stock_metadata
    'companyName', //stock_metadata
    'currency', //stock_metadata
    'Open', //stock_history_data
    'Close', //stock_history_data
    'Low', //stock_history_data
    'High', //stock_history_data
    'marketCap', //stock_financial_info
    'priceToBook', //stock_default_key_stats
  ];

  dataSource!: MatTableDataSource<any>;
  @ViewChild(MatSort) sort!: MatSort;
  @ViewChild(MatPaginator) paginator!: MatPaginator;

  constructor(
    private _stockService: StockService,
    private _liveAnnouncer: LiveAnnouncer
  ) {}

  announceSortChange(sortState: Sort) {
    if (sortState.direction) {
      this._liveAnnouncer.announce(`Sorted ${sortState.direction}ending`);
    } else {
      this._liveAnnouncer.announce('Sorting cleared');
    }
  }

  ngAfterViewInit(): void {
    this.dataSource.sort = this.sort;
    this.dataSource.paginator = this.paginator;
  }
  getData() {
    this._stockService.getAllPricingHistory().subscribe({
      next: (res) => {
        console.log(res);
        this.dataSource = new MatTableDataSource(res);
        this.dataSource.sort = this.sort;
        this.dataSource.paginator = this.paginator;
      },
      error: console.log,
    });
  }

  applyFilter(event: Event) {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();

    if (this.dataSource.paginator) {
      this.dataSource.paginator.firstPage();
    }
  }
}
