import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FilingComponent } from './filing.component';

describe('FilingComponent', () => {
  let component: FilingComponent;
  let fixture: ComponentFixture<FilingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [FilingComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(FilingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
