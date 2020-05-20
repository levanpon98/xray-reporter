import React, {Component} from 'react';
import Paper from '@material-ui/core/Paper';
import {
    Chart,
    BarSeries,
    Title,
    ArgumentAxis,
    ValueAxis,
    Tooltip
} from '@devexpress/dx-react-chart-material-ui';
import {Animation, Palette} from '@devexpress/dx-react-chart';
import {EventTracker, HoverState} from '@devexpress/dx-react-chart';
import {
    schemeCategory10
} from 'd3-scale-chromatic';


export default class PredictChart extends Component {
    constructor(props) {
        super(props);

        this.state = {
            hover: undefined,
            scheme: schemeCategory10
        };

        this.changeHover = hover => this.setState({hover});
    }

    render() {
        const {predict: chartData} = this.props;
        const {hover, scheme} = this.state
        return (
            <Paper>
                <Chart
                    data={chartData}

                >
                    <ArgumentAxis/>
                    <ValueAxis max={1}/>
                    <Palette scheme={scheme}/>
                    <BarSeries
                        valueField="probability"
                        argumentField="label"
                    />
                    <Title text="Predict Result"/>
                    <EventTracker/>
                    <HoverState hover={hover} onHoverChange={this.changeHover}/>
                    <Tooltip/>
                    <Animation/>
                </Chart>
            </Paper>
        );
    }
}