import React, {Fragment} from "react";
import PropTypes from "prop-types";
import {makeStyles, useTheme} from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import Grid from '@material-ui/core/Grid';
import PredictChart from "./PredictChart";
import Zoom from "react-medium-image-zoom";
import 'react-medium-image-zoom/dist/styles.css'

const useStyles = makeStyles((theme) => ({
    root: {
        minWidth: 200,
    },
    media: {
        height: 0,
        paddingTop: '78.1%'
    },

}));

function PredictCard(props) {
    const {image, predict} = props;
    const classes = useStyles();
    return (
        <Grid item xs={12}>
            <Grid
                container
                justify='center'
                spacing={3}
                alignItems='center'>
                <Grid item xs={12} sm={6}
                      alignContent='center'
                      justify='center'
                >

                    <Zoom zoomMargin={40}>
                        <img
                            alt="that wanaka tree"
                            src={`data:image/jpeg;base64,${image}`}
                            className="img"
                            style={{ width: '30em'}}
                        />
                    </Zoom>
                </Grid>
                <Grid item xs={12} sm={6}>
                    <PredictChart predict={predict}/>
                </Grid>
            </Grid>
        </Grid>
    );
}

PredictCard.propTypes = {
    image: PropTypes.string.isRequired,
    predict: PropTypes.array.isRequired
};

export default PredictCard;
