import { Component, OnInit, ViewChild } from "@angular/core";
import { MatDialog } from "@angular/material/dialog";
import { StockAddEditComponent } from "./stock-add-edit/stock-add-edit.component";
import { StockService } from "./services/stock.service";
import { MatPaginator } from "@angular/material/paginator";
import { MatSort } from "@angular/material/sort";
import { MatTableDataSource } from "@angular/material/table";
import { CoreService } from "./core/core.service";
import { StockInfoComponent } from "./stock-info/stock-info.component";
import { StockPriceComponent } from "./stock-price/stock-price.component";
import { StockPriceAllComponent } from "./stock-price-all/stock-price-all.component";
import { VisualComponent } from "./visual/visual.component";
import { BarComponent } from "./bar/bar.component";
import { PieComponent } from "./pie/pie.component";
import { FilingComponent } from "./filing/filing.component";

@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrl: "./app.component.scss",
})
export class AppComponent implements OnInit {
  title = "infs740_project_app";

  displayedColumns: string[] = [
    "symbol",
    "companyName",
    "sector",
    "industry",
    "country",
    "currency",
    "marketCap",
    "sharesOutstanding",
    "currentPrice",
    "totalRevenue",
    "enterpriseValue",
    "beta",
    "bookValue",
    "priceToBook",

    "action",
  ];
  dataSource!: MatTableDataSource<any>;

  @ViewChild(MatPaginator) paginator!: MatPaginator;
  @ViewChild(MatSort) sort!: MatSort;

  constructor(
    private _dialog: MatDialog,
    private _stockService: StockService,
    private _coreService: CoreService
  ) {}

  ngOnInit(): void {
    this.getFinanceData();
  }

  openAddEditFinanceForm() {
    const dialogRef = this._dialog.open(StockAddEditComponent);
    dialogRef.afterClosed().subscribe({
      next: (val) => {
        if (val) {
          this.getFinanceData();
        }
      },
    });
  }

  getFinanceData() {
    this._stockService.getFinanceDataList().subscribe({
      next: (res) => {
        // console.log(res);
        this.dataSource = new MatTableDataSource(res);
        this.dataSource.sort = this.sort;
        this.dataSource.paginator = this.paginator;
      },
      error: console.log,
    });
  }
  getFinanceOver1T() {
    this._stockService.getFinanceOver1T().subscribe({
      next: (res) => {
        // console.log(res);
        this._dialog.open(StockInfoComponent);
      },
      error: console.log,
    });
  }
  getHighestOpenPricingHistory() {
    this._stockService.getHighestOpenPricingHistory().subscribe({
      next: (res) => {
        this._dialog.open(StockPriceComponent);
      },
      error: console.log,
    });
  }
  getAllPricingHistory() {
    this._stockService.getAllPricingHistory().subscribe({
      next: (res) => {
        this._dialog.open(StockPriceAllComponent);
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

  deleteFinanceData(symbol: string) {
    this._stockService.deleteFinanceData(symbol).subscribe({
      next: (res) => {
        this._coreService.openSnackBar("Finance Data Deleted!", "done");
        this.getFinanceData();
      },
      error: console.log,
    });
  }

  openEditForm(data: any) {
    const dialogRef = this._dialog.open(StockAddEditComponent, {
      data,
    });

    dialogRef.afterClosed().subscribe({
      next: (val) => {
        if (val) {
          this.getFinanceData();
        }
      },
    });
  }
  displayBarChart() {
    // this._dialog.open(VisualComponent);
    this._dialog.open(BarComponent);
  }
  displayPieChart() {
    this._dialog.open(PieComponent);
  }
  displayFiling() {
    this._dialog.open(FilingComponent);
  }
}
