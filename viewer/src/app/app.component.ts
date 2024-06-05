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
import { ThemePalette } from "@angular/material/core";
import { DatePipe } from "@angular/common";

@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrl: "./app.component.scss",
  providers: [DatePipe],
})
export class AppComponent implements OnInit {
  title = "infs740_project_app";

  displayedColumns: string[] = [
    "cikNumber",
    "companyName",
    "formType",
    "filingDate",
    "fiscalYearEnd",
    "period",
    "acceptanceDatetime",
    // "accessionNumber",
    "fileNumber",
    "accessionNumber",
    // "link",

    // "action",
  ];
  dataSource!: MatTableDataSource<any>;

  @ViewChild(MatPaginator) paginator!: MatPaginator;
  @ViewChild(MatSort) sort!: MatSort;

  constructor(
    private _dialog: MatDialog,
    private _stockService: StockService,
    private _coreService: CoreService,
    private datePipe: DatePipe
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
    this._stockService.getFilingData().subscribe({
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

  parseTimestampToDate(timestamp: string): Date {
    timestamp = "20240604171853";
    const year = parseInt(timestamp.slice(0, 4), 10);
    const month = parseInt(timestamp.slice(4, 6), 10) - 1; // Months are zero-based in JavaScript
    const day = parseInt(timestamp.slice(6, 8), 10);
    const hours = parseInt(timestamp.slice(8, 10), 10);
    const minutes = parseInt(timestamp.slice(10, 12), 10);
    const seconds = parseInt(timestamp.slice(12, 14), 10);

    return new Date(year, month, day, hours, minutes, seconds);
  }
}
