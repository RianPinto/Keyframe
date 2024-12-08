import React, { useState } from "react";
import "../App.css";

const VideoInput = ({setSummary, setKeyFrames}) => {
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState("");
  const [uploadStatus, setUploadStatus] = useState(""); 

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoURL(URL.createObjectURL(file));
      setUploadStatus(""); 
    }
  };

  const handleClear = () => {
    setVideoFile(null)
    setVideoURL("")
    setUploadStatus("")
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    setUploadStatus("Uploading...");

    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await fetch("http://127.0.0.1:3000/video", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json(); // Parse the JSON response
        setUploadStatus("Success");
        setSummary(data.summary);
        setKeyFrames(data.keyframes)
      } else {
        throw new Error("Upload failed");
      }
    } catch (error) {
      console.error(error);
      setUploadStatus("Error");
    }
  };

  return (
    <div className="flex flex-col p-5 bg-gray-100 rounded-lg shadow-md h-[100%]">
      {console.log(uploadStatus)}
      {uploadStatus!="Uploading..." && uploadStatus!="Success" ? 
        <>
        <label
        htmlFor="video-upload"
        className="button-30 w-[30%] cursor-pointer px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-blue-200 focus:outline-none focus:ring focus:ring-blue-300"
      >
        Upload Video
      </label>
      <input
        id="video-upload"
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleVideoChange}
      /> </>:
      <button className="button-30 hover:bg-blue-200 w-[30%]" onClick={handleClear}>Clear Input</button> 
      }
      
      {videoFile && (
        <div className="w-[60%] mt-3">
          <video
            controls
            src={videoURL}
            className="w-full rounded-lg shadow-md"
            width={600}
          >
            Your browser does not support the video tag.
          </video>
          {uploadStatus=="Uploading..." || uploadStatus=="Success" ? <div className="text-2xl mt-4">{uploadStatus}</div> : <button className="button-30 hover:bg-blue-200  mt-4 w-[50%]" onClick={handleUpload}>Summarize</button>}
          
        </div>
      )}
    </div>
  );
};

export default VideoInput;
