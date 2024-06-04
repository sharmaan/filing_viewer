import { Component, Inject, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { CoreService } from '../core/core.service';
import { StockService } from '../services/stock.service';

@Component({
  selector: 'app-stock-add-edit',
  templateUrl: './stock-add-edit.component.html',
  styleUrl: './stock-add-edit.component.scss',
})
export class StockAddEditComponent implements OnInit {
  financeForm: FormGroup;

  industry: string[] = ['Consumer Electronics', 'Software - Infrastructure'];

  constructor(
    private _fb: FormBuilder,
    private _stockService: StockService,
    private _dialogRef: MatDialogRef<StockAddEditComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private _coreService: CoreService
  ) {
    this.financeForm = this._fb.group({
      symbol: '',
      companyName: '',
      sector: '',
      industry: '',
      country: '',
      currency: '',
      marketCap: '',
      sharesOutstanding: '',
      currentPrice: '',
      totalRevenue: '',
      enterpriseValue: '',
      beta: '',
      bookValue: '',
      priceToBook: '',
    });
  }

  ngOnInit(): void {
    this.financeForm.patchValue(this.data);
  }

  onFormSubmit() {
    if (this.financeForm.valid) {
      if (this.data) {
        this._stockService
          .updateFinanceData(this.data.symbol, this.financeForm.value)
          .subscribe({
            next: (val: any) => {
              this._coreService.openSnackBar('Finance Data detail updated!');
              this._dialogRef.close(true);
            },
            error: (err: any) => {
              console.error(err);
            },
          });
      } else {
        this._stockService.addFinanceData(this.financeForm.value).subscribe({
          next: (val: any) => {
            this._coreService.openSnackBar('Finance Data added successfully');
            this._dialogRef.close(true);
          },
          error: (err: any) => {
            console.error(err);
          },
        });
      }
    }
  }
}
