import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StockAddEditComponent } from './stock-add-edit.component';

describe('StockAddEditComponent', () => {
  let component: StockAddEditComponent;
  let fixture: ComponentFixture<StockAddEditComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [StockAddEditComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(StockAddEditComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
