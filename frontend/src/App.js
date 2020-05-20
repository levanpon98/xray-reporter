import React, {Fragment, Suspense, lazy} from "react";
import {MuiThemeProvider, CssBaseline} from "@material-ui/core";
import {BrowserRouter, Route, Switch} from "react-router-dom";
import theme from "./theme";
import GlobalStyles from "./GlobalStyles";
import * as serviceWorker from "./serviceWorker";
import Pace from "./shared/components/Pace";


const HomepageComponent = lazy(() => import("./homepage/components/Main"));

function App() {
    return (
        <BrowserRouter>
            <MuiThemeProvider theme={theme}>
                <CssBaseline/>
                <GlobalStyles/>
                <Pace color={theme.palette.primary.light}/>
                <Suspense fallback={<Fragment/>}>
                    <HomepageComponent/>
                </Suspense>
            </MuiThemeProvider>
        </BrowserRouter>
    );
}

serviceWorker.register();

export default App;
