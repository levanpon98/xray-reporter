import React, {Component} from "react";
import {DropzoneArea} from "material-ui-dropzone";

class DropzoneAreaExample extends Component {
    constructor(props) {
        super(props);
        this.state = {
            files: []
        };
    }

    handleChange(files) {
        this.setState({
            files: files
        }, this.uploadFile);
    }

    uploadFile = () => {
        const {files} = this.state
        const formData = new FormData();
        files.map((file, index) => {
            console.log(file)
            formData.append('images[]', file)
        })

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData,
            headers: {
                'content-type': 'multipart/form-data'
            }
        })
            .then(response => {
                console.log(response)
            }).catch(error => {
            console.log(error)
        })
    }

    render() {
        return <DropzoneArea
            onChange={this.handleChange.bind(this)}/>;
    }
}

export default DropzoneAreaExample;
