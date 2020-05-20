import React, {useContext} from "react";
import PropTypes from "prop-types";
import {Grid, Typography, isWidthUp, withWidth, withStyles} from "@material-ui/core";
import BuildIcon from "@material-ui/icons/Build";
import CalendarTodayIcon from "@material-ui/icons/CalendarToday";
import MeassageIcon from "@material-ui/icons/Message";
import calculateSpacing from "./calculateSpacing";
import PredictCard from "./PredictCard";
import {PredictContext} from "../../../context/PredictContext";
import DropzoneDialogExample from "../../../shared/components/DropzoneDialogExample";

const iconSize = 30;

const styles = theme => ({
    extraLargeButtonLabel: {
        fontSize: theme.typography.body1.fontSize,
        [theme.breakpoints.up("sm")]: {
            fontSize: theme.typography.h6.fontSize
        }
    },
    extraLargeButton: {
        paddingTop: theme.spacing(1.5),
        paddingBottom: theme.spacing(1.5),
        [theme.breakpoints.up("xs")]: {
            paddingTop: theme.spacing(1),
            paddingBottom: theme.spacing(1)
        },
        [theme.breakpoints.up("lg")]: {
            paddingTop: theme.spacing(2),
            paddingBottom: theme.spacing(2)
        }
    },
});

function ResultSection(props) {
    const things = useContext(PredictContext)
    const renderData = things => {
        // console.log(things.length)
        // console.log(things.data.length)
        if (things.data.length !== 0) {
            return things.data.map(thing => {
                return <PredictCard
                    image={thing.image}
                    predict={thing.predict}
                />
            })
        }
    }
    const {width, classes} = props;
    return (

        <div style={{backgroundColor: "#FFFFFF"}}>
            <div className="container-fluid lg-p-top">

                <div className="container-fluid">
                    <Grid
                        container
                        spacing={calculateSpacing(width)}
                        justify='center'
                        alignItems='center'>
                        <Grid item xs={12} sm={6}>
                            <DropzoneDialogExample
                                className={classes.extraLargeButton}
                                classes={{label: classes.extraLargeButtonLabel}}
                            />
                        </Grid>
                        {renderData(things)}
                    </Grid>
                </div>
            </div>
        </div>
    );
}

ResultSection.propTypes = {
    width: PropTypes.string.isRequired,
    classes: PropTypes.object
};
export default withWidth()(
    withStyles(styles, {withTheme: true})(ResultSection)
);
