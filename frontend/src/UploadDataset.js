import React, { useState } from "react";


function UploadDataset({ onUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);


  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };


  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    } else {
      alert("Please select a file before uploading.");
    }
  };


  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload and Train</button>
    </div>
  );
}


export default UploadDataset;


