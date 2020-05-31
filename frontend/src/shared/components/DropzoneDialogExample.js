import React, {Component} from "react";
import {DropzoneDialog} from "material-ui-dropzone";
import Button from "@material-ui/core/Button";
import {PredictContext} from "../../context/PredictContext";

export default class DropzoneDialogExample extends Component {
    static contextType = PredictContext

    constructor(props) {
        super(props);
        this.state = {
            open: false,
            files: []
        };
    }

    handleClose() {
        this.setState({
            open: false
        });
    }

    handleSave = (files) => {
        //Saving files to state for further use and closing Modal.
        this.setState({
            files: files,
            open: false,
        }, this.uploadFile);

    }

    uploadFile = () => {
        const {updateData} = this.context
        const {files} = this.state
        const formData = new FormData();
        const xhr = new XMLHttpRequest();

        Array.prototype.forEach.call(files, (file) => {
            formData.append('images[]', file)
        })

        xhr.open('POST', 'http://localhost:5000/predict', true)

        xhr.onload = () => {
            const data = JSON.parse(xhr.responseText);
            updateData(data.data)
        }
        xhr.onerror = () => {
            console.log('fail')
        }
        xhr.send(formData)
    }

    handleOpen() {
        this.setState({
            open: true
        });
    }

    render() {
        return (
            <div>
                <Button
                    variant="contained"
                    color="secondary"
                    fullWidth
                    className={this.props.className}
                    classes={this.props.classes}
                    onClick={this.handleOpen.bind(this)}>
                    Upload Image
                </Button>
                <DropzoneDialog
                    open={this.state.open}
                    onSave={this.handleSave.bind(this)}
                    acceptedFiles={["image/jpeg", "image/png", "image/bmp"]}
                    showPreviews={true}
                    cancelButtonText={"cancel"}
                    submitButtonText={"submit"}
                    maxFileSize={5000000}
                    onClose={this.handleClose.bind(this)}
                    showFileNamesInPreview={true}
                    filesLimit={5}
                />
            </div>
        );
    }
}
