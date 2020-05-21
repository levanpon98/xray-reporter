import React, {Fragment} from "react";
import PropTypes from "prop-types";
import {makeStyles, useTheme} from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import Grid from '@material-ui/core/Grid';
import PredictChart from "./PredictChart";
import Zoom from "react-medium-image-zoom";
import 'react-medium-image-zoom/dist/styles.css'
import clsx from 'clsx';
import CardHeader from '@material-ui/core/CardHeader';
import CardMedia from '@material-ui/core/CardMedia';
import CardContent from '@material-ui/core/CardContent';
import CardActions from '@material-ui/core/CardActions';
import Collapse from '@material-ui/core/Collapse';
import Avatar from '@material-ui/core/Avatar';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
import {red} from '@material-ui/core/colors';
import FavoriteIcon from '@material-ui/icons/Favorite';
import ShareIcon from '@material-ui/icons/Share';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import MoreVertIcon from '@material-ui/icons/MoreVert';
import calculateSpacing from "./calculateSpacing";
import CheckIcon from '@material-ui/icons/Check';
import TextareaAutosize from '@material-ui/core/TextareaAutosize';
import TextField from "@material-ui/core/TextField";

const useStyles = makeStyles((theme) => ({
    root: {
        maxWidth: 500,
    },
    media: {
        height: 0,
        paddingTop: '56.25%', // 16:9
    },
    expand: {
        transform: 'rotate(0deg)',
        marginLeft: 'auto',
        transition: theme.transitions.create('transform', {
            duration: theme.transitions.duration.shortest,
        }),
    },
    expandOpen: {
        transform: 'rotate(180deg)',
    },
    avatar: {
        backgroundColor: red[500],
    },
    textField: {
        marginLeft: theme.spacing(1),
        marginRight: theme.spacing(1),
        width: '25ch',
    },
}));

function PredictCard(props) {
    const {image, predict} = props;

    const classes = useStyles();
    const [expanded, setExpanded] = React.useState(false);
    const handleExpandClick = () => {
        setExpanded(!expanded);
    };
    return (
        <Grid item xs={6}>
            <Card style={{alignItems: "center", maxHeight: '500'}}>
                <CardContent>
                    <Grid container justify="center" alignItems='center'>
                        <Grid item xs={6}>
                            <Zoom zoomMargin={40}>
                                <img
                                    alt="that wanaka tree"
                                    src={`data:image/jpeg;base64,${image}`}
                                    className="img"
                                    style={{
                                        width: '100%',
                                    }}
                                />
                            </Zoom>
                        </Grid>
                    </Grid>
                </CardContent>
                <CardContent>
                    <Typography variant="body2" color="textSecondary" component="p">
                        {predict}
                    </Typography>
                </CardContent>
                <CardActions disableSpacing>
                    <IconButton aria-label="approve">
                        <CheckIcon/>
                    </IconButton>
                    <IconButton
                        className={clsx(classes.expand, {
                            [classes.expandOpen]: expanded,
                        })}
                        onClick={handleExpandClick}
                        aria-expanded={expanded}
                        aria-label="show more"
                    >
                        <ExpandMoreIcon/>
                    </IconButton>
                </CardActions>
                <Collapse in={expanded} timeout="auto" unmountOnExit>
                    <div>
                        <TextField
                            id="protocol1"
                            label="Protocol 1"
                            style={{margin: 8}}
                            placeholder="Protocol 1"
                            fullWidth
                            margin="normal"
                            InputLabelProps={{
                                shrink: true,
                            }}
                        />
                        <TextField
                            id="protocol2"
                            label="Protocol 2"
                            style={{margin: 8}}
                            placeholder="Protocol 2"
                            fullWidth
                            margin="normal"
                            InputLabelProps={{
                                shrink: true,
                            }}
                        />
                        <TextField
                            id="protocol3"
                            label="Protocol 3"
                            style={{margin: 8}}
                            placeholder="Protocol 3"
                            fullWidth
                            margin="normal"
                            InputLabelProps={{
                                shrink: true,
                            }}
                        />
                        <TextField
                            id="protocol4"
                            label="Protocol 4"
                            style={{margin: 8}}
                            placeholder="Protocol 4"
                            fullWidth
                            margin="normal"
                            InputLabelProps={{
                                shrink: true,
                            }}
                        />
                    </div>
                </Collapse>
            </Card>
        </Grid>
    );
}

PredictCard.propTypes = {
    image: PropTypes.string.isRequired,
    predict: PropTypes.array.isRequired
};

export default PredictCard;
