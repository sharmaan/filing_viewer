import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { StockPriceAllComponent } from './stock-price-all/stock-price-all.component';

const routes: Routes = [
  { path: '', redirectTo: '/listing', pathMatch: 'full' },
  {
    path: 'listing',
    component: StockPriceAllComponent,
    pathMatch: 'full',
  },
  {
    path: '',
    component: StockPriceAllComponent,
    pathMatch: 'full',
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
