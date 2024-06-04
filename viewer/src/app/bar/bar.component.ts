import { Component, OnInit } from '@angular/core';
import * as d3 from 'd3';
import { StockService } from '../services/stock.service';
//https://jsonbin.io/login
//https://blog.logrocket.com/data-visualization-angular-d3-js/
//json server api http://localhost:3000/bardata
//start from terminal npx json-server db.json
@Component({
  selector: 'app-bar',
  templateUrl: './bar.component.html',
  styleUrl: './bar.component.scss',
})
export class BarComponent implements OnInit {
  // private data = [
  //   { Framework: 'Vue', Stars: '166443', Released: '2014' },
  //   { Framework: 'React', Stars: '150793', Released: '2013' },
  //   { Framework: 'Angular', Stars: '62342', Released: '2016' },
  //   { Framework: 'Backbone', Stars: '27647', Released: '2010' },
  //   { Framework: 'Ember', Stars: '21471', Released: '2011' },
  // ];
  private svg: any;
  private margin = 50;
  private width = 750 - this.margin * 2;
  private height = 400 - this.margin * 2;

  data!: any;

  constructor(private _stockService: StockService) {}

  // ngOnInit(): void {
  //   this.createSvg();
  //   this.drawBars(this.data);
  // }
  // http://localhost:3000/bardata
  ngOnInit(): void {
    this.createSvg();
    //Fetch JSON from an external endpoint
    // type CharDataType = {
    //   Framework: string;
    //   Stars: number;
    //   Released: number;
    // };
    // d3.json('https://api.jsonbin.io/b/5eee6a5397cb753b4d149343').then(
    // d3.json('http://localhost:3000/bardata').then((data:any) => {
    //   const chartData = data as CharDataType[];
    //   this.drawBars(chartData);
    // });

    this._stockService.getDataToVisualize().subscribe((data) => {
      this.data = data;
      this.drawBars(this.data);
    });
    // this.drawBars(this.data);
  }

  private createSvg(): void {
    this.svg = d3
      .select('figure#bar')
      .append('svg')
      .attr('width', this.width + this.margin * 2)
      .attr('height', this.height + this.margin * 2)
      .append('g')
      .attr('transform', 'translate(' + this.margin + ',' + this.margin + ')');
  }

  private drawBars(data: any[]): void {
    // Create the X-axis band scale
    const x = d3
      .scaleBand()
      .range([0, this.width])
      .domain(data.map((d) => d.symbol))
      .padding(0.2);

    // Draw the X-axis on the DOM
    this.svg
      .append('g')
      .attr('transform', 'translate(0,' + this.height + ')')
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'translate(-10,0)rotate(-45)')
      .style('text-anchor', 'end');

    // Create the Y-axis band scale
    const y = d3.scaleLinear().domain([0, 2000]).range([this.height, 0]);

    // Draw the Y-axis on the DOM
    this.svg.append('g').call(d3.axisLeft(y));

    // Create and fill the bars
    this.svg
      .selectAll('bars')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', (d: any) => x(d.symbol))
      .attr('y', (d: any) => y(d.High))
      .attr('width', x.bandwidth())
      .attr('height', (d: any) => this.height - y(d.High))
      .attr('fill', '#d04a35');
  }
}
