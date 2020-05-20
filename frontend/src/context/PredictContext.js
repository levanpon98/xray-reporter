import React, {createContext, Component} from "react";

export const PredictContext = createContext();

class PredictContextProvider extends Component {
    state = {
        data: []
    }
    updateData = (data) => {
        this.setState({data: data})
    }

    render () {
        return (
            <PredictContext.Provider value={{...this.state, updateData: this.updateData}}>
                {this.props.children}
            </PredictContext.Provider>
        )
    }
}

export default PredictContextProvider;