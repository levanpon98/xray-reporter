import React, {Fragment, useEffect} from "react";
import PropTypes from "prop-types";
import HeadSection from "./HeadSection";
import ResultSection from "./ResultSection";
import PredictContextProvider from "../../../context/PredictContext";

function Home(props) {
    const {selectHome} = props;
    useEffect(() => {
        selectHome();
    }, [selectHome]);

    return (
        <Fragment>
            <PredictContextProvider>
                <HeadSection/>
                <ResultSection/>
            </PredictContextProvider>
        </Fragment>
    );
}

Home.propTypes = {
    selectHome: PropTypes.func.isRequired
};

export default Home;
