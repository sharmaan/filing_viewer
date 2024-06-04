import { AfterViewInit, Component, OnInit, ViewChild } from '@angular/core';
import { MatSort, Sort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';
import { StockService } from '../services/stock.service';
import { LiveAnnouncer } from '@angular/cdk/a11y';
import { MatPaginator } from '@angular/material/paginator';

@Component({
  selector: 'app-stock-info',
  templateUrl: './stock-info.component.html',
  styleUrl: './stock-info.component.scss',
})
export class StockInfoComponent implements OnInit, AfterViewInit {
  ngOnInit(): void {
    this.getData();
  }

  displayedColumns: string[] = [
    'symbol',
    'companyName',
    'sector',
    'industry',
    'country',
    'currentPrice',
    'marketCap',
    'sharesOutstanding',
    'totalRevenue',
    'bookValue',
  ];

  dataSource!: MatTableDataSource<any>;
  @ViewChild(MatSort) sort!: MatSort;
  @ViewChild(MatPaginator) paginator!: MatPaginator;

  constructor(
    private _stockService: StockService,
    private _liveAnnouncer: LiveAnnouncer
  ) {}

  announceSortChange(sortState: Sort) {
    // This example uses English messages. If your application supports
    // multiple language, you would internationalize these strings.
    // Furthermore, you can customize the message to add additional
    // details about the values being sorted.
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
    this._stockService.getFinanceOver1T().subscribe({
      next: (res) => {
        console.log(res);
        this.dataSource = new MatTableDataSource(res);
        this.dataSource.sort = this.sort;
        this.dataSource.paginator = this.paginator;
      },
      error: console.log,
    });
  }
}
