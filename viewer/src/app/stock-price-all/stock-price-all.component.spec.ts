import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StockPriceAllComponent } from './stock-price-all.component';

describe('StockPriceAllComponent', () => {
  let component: StockPriceAllComponent;
  let fixture: ComponentFixture<StockPriceAllComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [StockPriceAllComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(StockPriceAllComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
